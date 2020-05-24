import argparse
import cv2

from tqdm.auto import tqdm
from pathlib import Path

from utils.database import FaceRecognitionSystem
from utils.io_utils import read_yaml, read_image, save_image
from utils.draw_utils import draw_boxes, draw_landmarks

if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", help="path to video", type=str)
    parser.add_argument("--config", help="path to sdk config", type=str, default="config/config.yaml")
    parser.add_argument("--result_path", "-r", help="path to save processed video", default="demo/results")
    parser.add_argument(
        "--folders_path",
        "-fp",
        help="path to save folders with images",
        default="/home/d_barysevich/FaceRecognition/employees/images",
    )
    parser.add_argument(
        "--db_folder_path",
        "-dbp",
        help="path to save database",
        default="/home/d_barysevich/FaceRecognition/employees/database",
    )
    args = parser.parse_args()

    sdk_config = read_yaml(args.config)

    system = FaceRecognitionSystem(sdk_config)

    folders_path = args.folders_path
    db_folder_path = args.db_folder_path

    # create, save and load database initialized from folders containing user photos
    system.create_database_from_folders(folders_path)
    system.save_database(db_folder_path)
    system.load_database(db_folder_path)
    video_path = Path(args.path).expanduser().resolve()

    result_path = Path(args.result_path).expanduser().resolve()
    if not result_path.exists():
        result_path.mkdir(parents=True)

    path_to_save = Path(result_path, video_path.name).as_posix()

    """

    cap = cv2.VideoCapture(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)

    # video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        ret, frame = cap.read()
        """
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bboxes, landmarks, user_ids, similarities = system.sdk.recognize_faces(img)
        names = [system.get_user_name(uid) for uid in user_ids]

        draw_boxes(img, bboxes, name_tags=names, similarities=similarities)
        draw_landmarks(img, landmarks)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        """
        cv2.imshow("Webcam", frame)

        keyPressed = cv2.waitKey(1)

        if keyPressed == 27 or keyPressed == 1048603:
            break  # esc to quit

    cv2.destroyAllWindows()
    cap.release()
