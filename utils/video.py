## Convert video into frames
import cv2
import os
from IPython.display import HTML
from base64 import b64encode

def video_to_frame(video_path, dst):
    if video_path.endswith('.mp4'):
        capture = cv2.VideoCapture(video_path)
        frameNr = 0
        while (True):
            success, frame = capture.read()
            if success:
                if frameNr <= 9:
                    frame_name = '0000' + str(frameNr) + '.jpg'
                else:
                    frame_name = '000' + str(frameNr) + '.jpg'
                save_path = os.path.join(dst, frame_name)
                #print(f'saving {frame_name} to {save_path}...')
                cv2.imwrite(save_path, frame)
            else:
                break

            frameNr = frameNr+1
        capture.release()
    return frameNr

## Display the input video mask and the output video given the paths
def display_video(video1, video2):
  video1_path = video1
  video2_path = video2

  # Read video files
  video1 = open(video1_path, 'rb').read()
  video2 = open(video2_path, 'rb').read()

  # Encode videos to base64
  video1_base64 = b64encode(video1).decode()
  video2_base64 = b64encode(video2).decode()

  # Create data URLs
  video1_data_url = "data:video/mp4;base64," + video1_base64
  video2_data_url = "data:video/mp4;base64," + video2_base64

  gap_width = 33

  # Embed videos with a gap in the middle in an HTML container
  html_code = f"""
  <div style="display: flex;">
      <video width="800" height="600" controls>
          <source src="{video1_data_url}" type="video/mp4">
      </video>
      <div style="width: {gap_width}px;"></div>
      <video width="800" height="600" controls>
          <source src="{video2_data_url}" type="video/mp4">
      </video>
  </div>
  """

  # Display the HTML container
  HTML(html_code)