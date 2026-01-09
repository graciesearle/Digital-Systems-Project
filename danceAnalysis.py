import cv2
import mediapipe as mp
import numpy as np
import argparse

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


def extract_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
        else:
            keypoints = np.zeros(99)  # 33 keypoints * 3 (x, y, z)
        
        keypoints_list.append(keypoints)
    
    cap.release()
    return np.array(keypoints_list)


def compare_keypoints(kp1, kp2):
    min_frames = min(len(kp1), len(kp2))
    kp1, kp2 = kp1[:min_frames], kp2[:min_frames]
    
    similarities = [np.linalg.norm(kp1[i] - kp2[i]) for i in range(min_frames)]
    
    return np.mean(similarities)


def main(video1, video2):
    kp1 = extract_keypoints(video1)
    kp2 = extract_keypoints(video2)
    similarity_score = compare_keypoints(kp1, kp2)
    print(f"Pose similarity score: {similarity_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare dance poses in two videos.")
    parser.add_argument("video1", help="Path to first video")
    parser.add_argument("video2", help="Path to second video")
    args = parser.parse_args()
    
    main(args.video1, args.video2)
