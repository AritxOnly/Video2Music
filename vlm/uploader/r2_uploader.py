import boto3
import uuid
from vlm.uploader.uploader import UploaderInterface

class R2Uploader(UploaderInterface):
    def __init__(
        self,
        account_id: str,
        access_key_id: str,
        secret_access_key: str,
        bucket: str,
        public_base_url: str,
    ):
        self.bucket = bucket
        self.public_base_url = public_base_url.rstrip("/")

        self.s3 = boto3.client(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
        )

    def upload(self, path: str) -> str:
        # 生成唯一文件名
        key = f"{uuid.uuid4().hex}_{path.split('/')[-1]}"

        self.s3.upload_file(path, self.bucket, key)

        # 如果 bucket 已启用 Public Access，则可直接用公开 URL
        return f"{self.public_base_url}/{key}"