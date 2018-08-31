import numpy as np
import cv2

"""Video Show"""

def video_show():
  cap = cv2.VideoCapture(0)

  while(True):
      # Capture frame-by-frame
      ret, frame = cap.read()

      # Our operations on the frame come here
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

      # Display the resulting frame
      cv2.imshow('frame',gray)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

  # When everything done, release the capture
  cap.release()
  cv2.destroyAllWindows()


"""Video Play"""

def video_play():

  cap = cv2.VideoCapture('ball_video.avi')

  while(cap.isOpened()):
      ret, frame = cap.read()

      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

      cv2.imshow('frame',gray)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

  cap.release()
  cv2.destroyAllWindows()


"""Video Save"""
def video_save():

  cap = cv2.VideoCapture(0)
  ret, frame = cap.read()
  height, width = frame.shape[:2]

  # Define the codec and create VideoWriter object
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  out = cv2.VideoWriter('ball_video.avi', fourcc, 20.0, (width, height))

  while(cap.isOpened()):
      ret, frame = cap.read()
      if ret==True:

          # write the flipped frame
          out.write(frame)

          cv2.imshow('frame',frame)
          if cv2.waitKey(1) & 0xFF == ord('q'):
              break
      else:
          break

  # Release everything if job is finished
  cap.release()
  out.release()
  cv2.destroyAllWindows()


if __name__=="__main__":
  video_save()
