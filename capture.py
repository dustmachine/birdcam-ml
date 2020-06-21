from picamera import PiCamera
from time import sleep
from datetime import datetime
from pathlib import Path

camera = PiCamera()
# if it is upside down
camera.rotation = 180
camera.resolution = (1440, 960)

p = Path('/var/www/html/capture.jpg')

while True:
  date_str = datetime.now().strftime('%Y%m%d-%H%M%S')
  filename = '/var/www/html/%s.jpg' % date_str
  camera.capture(filename)
  # make a link
  p.unlink()
  p.symlink_to(filename)
  sleep(5)

