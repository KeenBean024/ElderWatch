import os
import cv2
import time
import torch
import argparse
import numpy as np

from Detection.Utils import ResizePadding
from CameraLoader import CamLoader, CamLoader_Q
from DetectorLoader import YOLOv10Detector
from GestureDetectoLoader import GestureDetector

from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single

from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG

source = './'

# Path to gesture detection model
gesture_model_path = './Models/yolo-gesture/YOLOv10n_gestures.pt'


def preproc(image):
    """preprocess function for CameraLoader.
    """
    image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))

if __name__ == '__main__':
    par = argparse.ArgumentParser(description='Human Fall Detection with Gesture Recognition.')
    par.add_argument('-C', '--camera', default=source,
                     help='Source of camera or video file path.')
    par.add_argument('--detection_input_size', type=int, default=640,
                     help='Size of input in detection model in square must be divisible by 32 (int).')
    par.add_argument('--pose_input_size', type=str, default='224x160',
                     help='Size of input in pose model must be divisible by 32 (h, w)')
    par.add_argument('--pose_backbone', type=str, default='resnet50',
                     help='Backbone model for SPPE FastPose model.')
    par.add_argument('--show_detected', default=False, action='store_true',
                     help='Show all bounding box from detection.')
    par.add_argument('--show_skeleton', default=True, action='store_true',
                     help='Show skeleton pose.')
    par.add_argument('--save_out', type=str, default='output_with_gestures_2.avi',
                     help='Save display to video file.')
    par.add_argument('--device', type=str, default='cuda',
                     help='Device to run model on cpu or cuda.')
    par.add_argument('--gesture_model', type=str, default=gesture_model_path,
                     help='Path to the YOLO gesture detection model.')
    par.add_argument('--enable_gesture', default=True, action='store_true',
                     help='Enable gesture detection when fall is detected.')
    args = par.parse_args()

    device = args.device

    # DETECTION MODEL - Using YOLOv10-n for person detection
    inp_dets = args.detection_input_size
    detect_model = YOLOv10Detector(inp_dets, device=device)

    # POSE MODEL for skeleton tracking
    inp_pose = args.pose_input_size.split('x')
    inp_pose = (int(inp_pose[0]), int(inp_pose[1]))
    pose_model = SPPE_FastPose(args.pose_backbone, inp_pose[0], inp_pose[1], device=device)

    # Gesture detection model
    gesture_detector = GestureDetector(args.gesture_model, device=device)

    # Tracker for person tracking
    max_age = 30
    tracker = Tracker(max_age=max_age, n_init=3)

    # Actions Estimate for fall detection
    action_model = TSSTG()

    resize_fn = ResizePadding(inp_dets, inp_dets)

    cam_source = args.camera
    if type(cam_source) is str and os.path.isfile(cam_source):
        # Use loader thread with Q for video file.
        cam = CamLoader_Q(cam_source, queue_size=1000, preprocess=preproc).start()
    else:
        # Use normal thread loader for webcam.
        cam = CamLoader(int(cam_source) if cam_source.isdigit() else cam_source,
                      preprocess=preproc).start()

    # Create video writer for output
    output_size = (inp_dets * 2, inp_dets * 2)
    codec = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(args.save_out, codec, 30, output_size)
    
    print(f"Processing video... Output will be saved to: {args.save_out}")
    print(f"Using YOLOv10-n for person detection on {device}")
    if args.enable_gesture:
        print(f"Gesture detection enabled - will analyze gestures during falls")
    
    fall_detected = False
    fps_time = 0
    f = 0
    try:
        while cam.grabbed():
            f += 1
            frame = cam.getitem()
            image = frame.copy()

            # Detect humans bbox in the frame with YOLOv10 model
            detected = detect_model.detect(frame, need_resize=False, expand_bb=10)

            # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
            tracker.predict()
            
            # Merge two source of predicted bbox together.
            for track in tracker.tracks:
                # Create a tensor with the same format as the detected tensor (matching dimensions)
                det = torch.tensor([track.to_tlbr().tolist() + [0.5]], dtype=torch.float32)
                detected = torch.cat([detected, det], dim=0) if detected is not None else det

            detections = []  # List of Detections object for tracking.
            if detected is not None:
                # Predict skeleton pose of each bboxs.
                poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])

                # Create Detections object.
                detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                      np.concatenate((ps['keypoints'].numpy(),
                                                      ps['kp_score'].numpy()), axis=1),
                                      ps['kp_score'].mean().numpy()) for ps in poses]

                # VISUALIZE.
                if args.show_detected:
                    for bb in detected[:, 0:5]:
                        frame = cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 0, 255), 1)

            # Update tracks by matching each track information of current and previous frame or
            # create a new track if no matched.
            tracker.update(detections)

            # Flag to track if we found a fall in this frame
            fall_bboxes = []

            # Predict Actions of each track.
            for i, track in enumerate(tracker.tracks):
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                bbox = track.to_tlbr().astype(int)
                center = track.get_center().astype(int)

                action = 'pending..'
                clr = (0, 255, 0)
                # Use 30 frames time-steps to prediction.
                if len(track.keypoints_list) == 30:
                    pts = np.array(track.keypoints_list, dtype=np.float32)
                    out = action_model.predict(pts, frame.shape[:2])
                    action_name = action_model.class_names[out[0].argmax()]
                    action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
                    
                    # Check if action is 'Fall Down'
                    if action_name == 'Fall Down':
                        fall_detected = True
                        fall_bboxes.append(bbox)
                        clr = (255, 0, 0)  # Red for fall
                    elif action_name == 'Lying Down':
                        clr = (255, 200, 0)  # Orange for lying

                # VISUALIZE.
                if track.time_since_update == 0:
                    if args.show_skeleton:
                        frame = draw_single(frame, track.keypoints_list[-1])
                    frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                    frame = cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX,
                                      0.4, (255, 0, 0), 2)
                    frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,
                                      0.4, clr, 1)

            # If fall detected and gesture detection is enabled, run gesture detection
            if fall_detected and args.enable_gesture and gesture_detector.model is not None:
                # Process gesture detection on the frame
                gesture_results = gesture_detector.detect(frame)
                
                if gesture_results:
                    # Draw all gesture detections
                    gesture_frame = gesture_results[0].plot()
                    
                    # Extract detection info to overlay on our output
                    boxes = gesture_results[0].boxes
                    for i in range(len(boxes)):
                        # Get box coordinates and label
                        box = boxes.xyxy[i].tolist()  # [x1, y1, x2, y2]
                        conf = boxes.conf[i].item()
                        cls_id = int(boxes.cls[i].item())
                        gesture_name = gesture_results[0].names[cls_id]
                        
                        # Draw a special gesture box on our frame
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow for gestures
                        
                        # Add gesture label
                        label = f"{gesture_name}: {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5, (0, 255, 255), 2)
                        
                        # Add a large warning text when both fall and gesture are detected
                        cv2.putText(frame, "ALERT: FALL + GESTURE DETECTED", 
                                  (50, 50), cv2.FONT_HERSHEY_DUPLEX,
                                  1.0, (0, 0, 255), 2)

            # Prepare frame for saving
            frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)
            frame = cv2.putText(frame, '%d, FPS: %f' % (f, 1.0 / (time.time() - fps_time)),
                              (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            fps_time = time.time()

            # Save frame to video
            writer.write(frame)
            
            # Print progress periodically
            if f % 100 == 0:
                print(f"Processed {f} frames...")
                
    except KeyboardInterrupt:
        print("Processing interrupted by user")
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()  # Print the full stack trace for debugging
    finally:
        # Clear resources
        print(f"Processing complete. {f} frames processed.")
        print(f"Output saved to: {args.save_out}")
        cam.stop()
        writer.release()