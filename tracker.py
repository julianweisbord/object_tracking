import cv2
import sys
import detector


class ObjectTracker():
    def __init__(self, init_detector):
        # Get object detector
        self.video = None
        self.tracker = None
        self.detector = init_detector

    def load_video(self, video_path):
        self.video = cv2.VideoCapture(video_path)

    def init_tracker(self, tracker_name):
        if tracker_name == "csrt":
            self.tracker = cv2.TrackerCSRT_create()

    def select_obj(self, img):
        if self.tracker:
            bbox = cv2.selectROI(img, False)
            return bbox
        else:
            print("No tracker")
            return None

    def display_new_bbox(self, img, bbox):
        print("bbox: ", bbox)
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (255,0,0), 2)
        return img

    def track(self, use_detector=False, show_tracking=False):
        # take a bbox from a detection or selected by User and take in a video,
        # track the object in the video and
        # every n frames, run object detection if selected
        if not self.video:
            print("No video supplied")
            exit()

        ret, frame = self.video.read()
        for i in range(200):
            ret, frame = self.video.read()
        if use_detector:
            pass
        else:
            selected_bbox = self.select_obj(frame)
            print("After selected")
        tracker_ret = self.tracker.init(frame, selected_bbox)

        while(self.video.isOpened()):

            ret, frame = self.video.read()

            if ret:
                tracker_ret, bbox = self.tracker.update(frame)
                bbox_img = self.display_new_bbox(frame, bbox)

                if show_tracking and tracker_ret:
                    cv2.imshow('frame: ', bbox_img)
            else:
                sys.exit()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video.release()
        cv2.destroyAllWindows()


def main():
    ssd_mobilenetv2 = detector.load_net()
    csrt = ObjectTracker(ssd_mobilenetv2)

    csrt.load_video("chaplin.mp4")
    csrt.init_tracker("csrt")
    csrt.track(use_detector=False, show_tracking=True)


if __name__ == '__main__':
    main()
