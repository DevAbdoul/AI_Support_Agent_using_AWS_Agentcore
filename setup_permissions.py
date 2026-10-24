"""
Grant the AgentCore runtime execution role access to the Knowledge Base,
Memory, Browser, Code Interpreter and Gateway used by main.py.
Safe to rerun: it overwrites the same inline policy each time.
"""

import json
import boto3

REGION = "us-east-1"
KB_ID = "GTOHPLGCMN"
MEMORY_ID = "CustomerSupportMemory-8gS2VVA5kw"
ROLE_NAME = "AmazonBedrockAgentCoreSDKRuntime-us-east-1-d0cb11a8b3"
POLICY_NAME = "CSAIAgentResourceAccess"

account_id = boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]

policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "KnowledgeBaseRetrieve",
            "Effect": "Allow",
            "Action": ["bedrock:Retrieve"],
            "Resource": f"arn:aws:bedrock:{REGION}:{account_id}:knowledge-base/{KB_ID}",
        },
        {
            "Sid": "ModelInvoke",
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream",
            ],
            "Resource": "*",
        },
        {
            "Sid": "MemoryAccess",
            "Effect": "Allow",
            "Action": [
                "bedrock-agentcore:GetMemory",
                "bedrock-agentcore:CreateEvent",
                "bedrock-agentcore:GetEvent",
                "bedrock-agentcore:ListEvents",
                "bedrock-agentcore:ListSessions",
                "bedrock-agentcore:ListActors",
                "bedrock-agentcore:RetrieveMemoryRecords",
                "bedrock-agentcore:ListMemoryRecords",
                "bedrock-agentcore:GetMemoryRecord",
            ],
            "Resource": [
                f"arn:aws:bedrock-agentcore:{REGION}:{account_id}:memory/{MEMORY_ID}",
                f"arn:aws:bedrock-agentcore:{REGION}:{account_id}:memory/{MEMORY_ID}/*",
            ],
        },
        {
            "Sid": "BrowserAndCodeInterpreter",
            "Effect": "Allow",
            "Action": [
                "bedrock-agentcore:StartBrowserSession",
                "bedrock-agentcore:StopBrowserSession",
                "bedrock-agentcore:GetBrowserSession",
                "bedrock-agentcore:ListBrowserSessions",
                "bedrock-agentcore:ConnectBrowserAutomationStream",
                "bedrock-agentcore:ConnectBrowserLiveViewStream",
                "bedrock-agentcore:UpdateBrowserStream",
                "bedrock-agentcore:StartCodeInterpreterSession",
                "bedrock-agentcore:StopCodeInterpreterSession",
                "bedrock-agentcore:GetCodeInterpreterSession",
                "bedrock-agentcore:ListCodeInterpreterSessions",
                "bedrock-agentcore:InvokeCodeInterpreter",
            ],
            "Resource": "*",
        },
        {
            "Sid": "GatewayInvoke",
            "Effect": "Allow",
            "Action": ["bedrock-agentcore:InvokeGateway"],
            "Resource": "*",
        },
    ],
}

iam = boto3.client("iam", region_name=REGION)
iam.put_role_policy(
    RoleName=ROLE_NAME,
    PolicyName=POLICY_NAME,
    PolicyDocument=json.dumps(policy),
)
print(f"Policy '{POLICY_NAME}' attached to role '{ROLE_NAME}' (account {account_id}).")
print("Wait 1-2 minutes for permissions to take effect, then run the tests.")
