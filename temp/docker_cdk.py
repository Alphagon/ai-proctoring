from aws_cdk import aws_ecr as ecr
from aws_cdk.aws_ecr_assets import DockerImageAsset
from aws_cdk import Stack, App, CfnOutput, RemovalPolicy
from aws_cdk import aws_iam as iam
from aws_cdk.aws_s3 import Bucket
from pathlib import Path

def create_ecr_repository(stack: Stack) -> ecr.Repository:
    """Creates an ECR repository and returns the repository object."""
    ecr_repository = ecr.Repository(stack, "MyEcrRepository", 
                                    repository_name="test-repo",
                                    removal_policy=RemovalPolicy.DESTROY)  # Optional: to automatically delete the repository when the stack is deleted
    return ecr_repository

def build_docker_image(stack: Stack, ecr_repository: ecr.Repository) -> DockerImageAsset:
    """Builds a Docker image from the specified directory and returns the DockerImageAsset."""
    docker_image_asset = DockerImageAsset(stack, "MyDockerImage", directory=str(Path(__file__).parent),)
    
    # Allow the Docker image to be pulled by this account
    ecr_repository.add_lifecycle_rule(description="Retain only 5 images",
                                      max_image_count=5)

    # Create an IAM role to use as a grantee
    role = iam.Role(stack, "alphagon",
                    assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"))  # or any other service that needs access

    # Grant the role permission to pull/push images to the ECR repository
    ecr_repository.grant_pull_push(role)

    # This ensures the Docker image is pushed to the specific ECR repository
    docker_image_asset.image_uri  # This triggers the build and push of the Docker image to ECR

    CfnOutput(stack, "ECRRepoUri", value=ecr_repository.repository_uri)

def main():
    app = App()
    stack = Stack(app, "DockerImageToEcrStack")
    
    # Create resources
    ecr_repository = create_ecr_repository(stack)
    build_docker_image(stack, ecr_repository)

    app.synth()

#if __name__ == "__main__":
#    main()

