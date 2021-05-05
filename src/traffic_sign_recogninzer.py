import cv2


class traffic_sign_recogninzer(object):
    sift = None
    sign_label = [
        'stop',
        'pedestrian crossing',
        'speed limit: 25',
        'speed limit: 35'
    ]

    sign_input_paths = [
        '/Users/hbojja/uiuc/CS445-CP/FinalProject/input/traffic_signs/stop_sign_4.jpeg',
        '/Users/hbojja/uiuc/CS445-CP/FinalProject/input/traffic_signs/ped_crossing_3.jpeg',
        '/Users/hbojja/uiuc/CS445-CP/FinalProject/input/traffic_signs/speed_limit_25_1.jpeg',
        '/Users/hbojja/uiuc/CS445-CP/FinalProject/input/traffic_signs/speed_limit_35.jpeg'
    ]

    sign_min_match_count = {
        sign_label[0]: 3,
        sign_label[1]: 2,
        sign_label[2]: 3,
        sign_label[3]: 2
    }

    sign_kps = []
    sign_descriptors = []

    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create()
        for sign_path in self.sign_input_paths:
            kp, des = self.prepare_sign_descriptors(cv2.imread(sign_path))
            self.sign_kps.append(kp)
            self.sign_descriptors.append(des)

    def recognize_signs(self, frame):
        return

    def convert_to_grey_scale_and_detectkps(self, image):
        image_grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        kp, des = self.sift.detectAndCompute(image_grey, None)

        return kp, des

    def match_des(self, template_des, frame_des):
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(template_des, frame_des, k=2)

        fair_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                fair_matches.append([m])

        return fair_matches, len(fair_matches) / len(template_des)

    def realtime_template_matcher(self, input_frame):
        frame_kp, frame_des = self.sift.detectAndCompute(input_frame, None)

        matched_decs_counts = []
        for idx in range(len(self.sign_descriptors)):
            template_des = self.sign_descriptors[idx]
            fair_maches, template_mpercntage = self.match_des(template_des, frame_des)
            matched_decs_counts.append(len(fair_maches))

            print(self.sign_label[idx], len(fair_maches))

        print('----------')
        max_count = max(matched_decs_counts)
        highest_match_index = matched_decs_counts.index(max_count)
        rec_label = self.sign_label[highest_match_index]

        if max_count > self.sign_min_match_count[rec_label]:
            return rec_label, max_count
        else:
            return '', 0

    def prepare_sign_descriptors(self, image):
        template_kp, template_des = self.convert_to_grey_scale_and_detectkps(image)

        return template_kp, template_des
