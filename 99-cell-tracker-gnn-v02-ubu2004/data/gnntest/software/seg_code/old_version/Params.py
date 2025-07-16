import DataHandeling
import os
from datetime import datetime
import Networks as Nets

__author__ = 'arbellea@post.bgu.ac.il'

# ROOT_SAVE_DIR = '/Users/aarbelle/Documents/DeepCellSegOut'
# ROOT_DATA_DIR3D = '/Users/aarbelle/Documents/CellTrackingChallenge/Training'

#
ROOT_DATA_DIR = '/media/rrtammyfs/labDatabase/CellTrackingChallenge/Training/'
ROOT_SAVE_DIR = '/media/rrtammyfs/Users/arbellea/LSTMUnet'
# ROOT_DATA_DIR = '/media/efs/CellTrackingChallenge/Training/'
# ROOT_SAVE_DIR = '/media/efs/Outputs/LSTMUnet'


class ParamsBase(object):
    aws = False
    input_depth = 0
    add_tra_output = False
    send_email = True
    email_username = 'bgumedicalimaginglab@gmail.com'
    email_password = "xmvipxhcpxvbeamc"
    receiver_email = 'arbellea@post.bgu.ac.il'

    def _override_params_(self, params_dict: dict):
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        this_dict = self.__class__.__dict__.keys()
        for key, val in params_dict.items():
            if key not in this_dict:
                print('Warning!: Parameter:{} not in defualt parameters'.format(key))
            setattr(self, key, val)

    pass

class CTCInferenceParams(ParamsBase):

    gpu_id = 0  # for CPU ise -1 otherwise gpu id
    seq = 2
    model_path = '/media/rrtammyfs/Users/arbellea/LSTMUnet/FromAWS/LSTMUNet2D/PhC-C2DL-PSC'
    output_path = '/media/rrtammyfs/labDatabase/CellTrackingChallenge/Test/PhC-C2DL-PSC/{:02d}_RES'.format(seq)
    sequence_path = '/media/rrtammyfs/labDatabase/CellTrackingChallenge/Test/PhC-C2DL-PSC/{:02d}'.format(seq)
    filename_format = 't*.tif'  # default format for CTC
    edge_thresh = 0.1
    data_reader = DataHandeling.CTCInferenceReader
    FOV = 25
    data_format = 'NCHW'  # 'NCHW' or 'NHWC'
    min_cell_size = 10
    max_cell_size = 10000
    edge_dist = 2
    pre_sequence_frames = 4
    centers_sigmoid_threshold = 0.2
    min_center_size = 1

    # ---------Debugging---------

    dry_run = False
    save_intermediate = True
    save_intermediate_path = os.path.join(model_path, 'outputs', os.path.basename(output_path))

    def __init__(self, params_dict: dict = None):

        if params_dict is not None:
            self._override_params_(params_dict)
        if isinstance(self.gpu_id, list):

            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)[1:-1]
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        self.channel_axis = 1 if self.data_format == 'NCHW' else 3
        if not self.dry_run:
            os.makedirs(self.output_path, exist_ok=True)
            if self.save_intermediate:
                now_string = datetime.now().strftime('%Y-%m-%d_%H%M%S')
                self.save_intermediate_path = os.path.join(self.save_intermediate_path, now_string)
                self.save_intermediate_vis_path = os.path.join(self.save_intermediate_path, 'Softmax')
                self.save_intermediate_label_path = os.path.join(self.save_intermediate_path, 'Labels')
                self.save_intermediate_label_color_path = os.path.join(self.save_intermediate_path, 'Color')
                self.save_intermediate_centers_path = os.path.join(self.save_intermediate_path, 'Centers')
                self.save_intermediate_centers_hard_path = os.path.join(self.save_intermediate_path, 'CentersHard')
                self.save_intermediate_contour_path = os.path.join(self.save_intermediate_path, 'Contours')
                os.makedirs(self.save_intermediate_path, exist_ok=True)
                os.makedirs(self.save_intermediate_vis_path, exist_ok=True)
                os.makedirs(self.save_intermediate_label_path, exist_ok=True)
                os.makedirs(self.save_intermediate_label_color_path, exist_ok=True)
                os.makedirs(self.save_intermediate_centers_path, exist_ok=True)
                os.makedirs(self.save_intermediate_contour_path, exist_ok=True)
                os.makedirs(self.save_intermediate_centers_hard_path, exist_ok=True)

class CTCInferenceParams3DSlice(ParamsBase):

    gpu_id = 1  # for CPU ise -1 otherwise gpu id
    seq = 2
    model_path = '/media/rrtammyfs/Users/arbellea/LSTMUnet/FromAWS/LSTMUNet3DSlice/Fluo-C3DL-MDA231'
    output_path = '/media/rrtammyfs/labDatabase/CellTrackingChallenge/Test/Fluo-C3DL-MDA231/{:02d}_RES'.format(seq)
    sequence_path = '/media/rrtammyfs/labDatabase/CellTrackingChallenge/Test/Fluo-C3DL-MDA231/{:02d}'.format(seq)
    filename_format = 't*.tif'  # default format for CTC

    data_reader = DataHandeling.CTCInferenceReader3DSlice
    FOV = 10
    data_format = 'NCHW'  # 'NCHW' or 'NHWC'
    min_cell_size = 100
    max_cell_size = 100000
    edge_dist = 2
    pre_sequence_frames = 0
    centers_sigmoid_threshold = 0.5
    min_center_size = 10
    edge_thresh = 0.5
    one_object = False

    # ---------Debugging---------

    dry_run = False
    save_intermediate = True
    save_intermediate_path = os.path.join(model_path, 'Outputs', os.path.basename(sequence_path))

    def __init__(self, params_dict: dict = None):

        if params_dict is not None:
            self._override_params_(params_dict)
        if isinstance(self.gpu_id, list):

            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)[1:-1]
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        self.channel_axis = 1 if self.data_format == 'NCHW' else 3
        if not self.dry_run:
            os.makedirs(self.output_path, exist_ok=True)
            if self.save_intermediate:
                now_string = datetime.now().strftime('%Y-%m-%d_%H%M%S')
                self.save_intermediate_path = os.path.join(self.save_intermediate_path, now_string)
                self.save_intermediate_vis_path = os.path.join(self.save_intermediate_path, 'Softmax')
                self.save_intermediate_label_path = os.path.join(self.save_intermediate_path, 'Labels')
                self.save_intermediate_contour_path = os.path.join(self.save_intermediate_path, 'Contours')
                self.save_intermediate_centers_path = os.path.join(self.save_intermediate_path, 'Centers')
                self.save_intermediate_centers_hard_path = os.path.join(self.save_intermediate_path, 'CentersHard')
                os.makedirs(self.save_intermediate_path, exist_ok=True)
                os.makedirs(self.save_intermediate_vis_path, exist_ok=True)
                os.makedirs(self.save_intermediate_label_path, exist_ok=True)
                os.makedirs(self.save_intermediate_contour_path, exist_ok=True)
                os.makedirs(self.save_intermediate_centers_path, exist_ok=True)
                os.makedirs(self.save_intermediate_centers_hard_path, exist_ok=True)




