import cv2


class traffic_sign_recogninzer_filter_individual(object):
    sift = None
    template_kp = None
    template_des = None
    match_threshold = None
    
    def __init__(self, sign_path, m_threshold):
        print('initialializing...')
        
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.match_threshold = m_threshold
        self.template_kp, self.template_des = self.prepare_sign_descriptors(cv2.imread(sign_path))
        

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
        
        fair_maches, template_mpercntage = self.match_des(self.template_des, frame_des)
        
        print(len(fair_maches),self.match_threshold, len(fair_maches) > self.match_threshold)
        
        if len(fair_maches) > self.match_threshold:
            return True
        else:
            return False

    def prepare_sign_descriptors(self, image):
        template_kp, template_des = self.convert_to_grey_scale_and_detectkps(image)

        return template_kp, template_des
