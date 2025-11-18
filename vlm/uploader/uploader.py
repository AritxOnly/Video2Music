from typing import Protocol, runtime_checkable

@runtime_checkable
class UploaderInterface(Protocol):
    def upload(path: str) -> str:
        ...
        
def get_uploader(opt: str = 'default') -> UploaderInterface:
    match(opt):
        case 'default' | 'r2':
            from vlm.uploader.r2_uploader import R2Uploader
            return R2Uploader(
                account_id="xxxxxxx",
                access_key_id="xxxxxxx",
                secret_access_key="xxxxxxx",
                bucket="my-bucket",
                public_base_url="https://pub-xxxxxx.r2.dev",
            )
        case _:
            raise ValueError(f"Unknown uploader: {opt}")