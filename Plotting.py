import scipy.io as scio #for loading raw EEG


import numpy as np
import mne 
import torch
import matplotlib
import matplotlib.pyplot as plt
import PIL

#from torcheeg.utils import plot_signal
#from torcheeg.constants.emotion_recognition import seed

class PlotEEG():
    def __init__(self):
        # self.patch_size = patch_size
        super().__init__()

    sampling_rate = 200
    
    SEED_CHANNEL_LIST = [
        'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4',
        'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
        'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3',
        'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ',
        'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
        'CB1', 'O1', 'OZ', 'O2', 'CB2'
    ]
    
    mne.set_log_level('CRITICAL')
    
    default_montage = mne.channels.make_standard_montage('standard_1020')
    
    def __call__(self,
                 tensor,
                 channel_list = SEED_CHANNEL_LIST,
                 sampling_rate = sampling_rate,
                 montage = default_montage):
        r'''
        Plot signal values of the input raw EEG as image.
    
        .. code-block:: python
    
            eeg = torch.randn(32, 128)
            img = plot_signal(eeg,
                              channel_list=DEAP_CHANNEL_LIST,
                              sampling_rate=128)
            # If using jupyter, the output image will be drawn on notebooks.
    
        .. image:: _static/plot_signal.png
            :alt: The output image of plot_signal
            :align: center
    
        |       |
    
        Args:
                tensor (torch.Tensor): The input EEG signal, the shape should be [number of channels, number of data points].
            channel_list (list): The channel name lists corresponding to the input EEG signal. If the dataset in TorchEEG is used, please refer to the CHANNEL_LIST related constants in the :obj:`torcheeg.constants` module.
            sampling_rate (int): Sample rate of the data.
            montage (any): Channel positions and digitization points defined in obj:`mne`. (default: :obj:`mne.channels.make_standard_montage('standard_1020')`)
    
        Returns:
            np.ndarray: The output image in the form of :obj:`np.ndarray`.
        '''
    
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor) 
        if tensor.dim() > 2:
            tensor = tensor.squeeze()
                     
        ch_types = ['misc'] * len(channel_list)
        info = mne.create_info(ch_names=channel_list,
                               ch_types=ch_types,
                               sfreq=sampling_rate)
    
        epochs = mne.io.RawArray(tensor.detach().cpu().numpy(), info)
        epochs.set_montage(montage, match_alias=True, on_missing='ignore')
        epochs.plot(show_scrollbars=True, show_scalebars=True, block=True)
        plt.show()
        return epochs

        #https://github.com/torcheeg/torcheeg/blob/main/torcheeg/datasets/constants/emotion_recognition/seed.py