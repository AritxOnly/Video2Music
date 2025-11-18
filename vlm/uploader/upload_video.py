from vlm.model import VideoInput
from vlm.uploader.uploader import get_uploader

def get_video_url(video: VideoInput) -> str:
    if video.url:
        return video.url
    elif video.path:
        # Do Upload
        url = upload_video(video.path)
        return url
    else:
        raise RuntimeError('No active video status')
    
def upload_video(path, opt='default') -> str:
    uploader = get_uploader
    return ''