# Clone file code


```python
!git clone https://github.com/duongstudent/CV_Vinbigdata.git
```

    Cloning into 'CV_Vinbigdata'...
    remote: Enumerating objects: 66, done.[K
    remote: Counting objects: 100% (66/66), done.[K
    remote: Compressing objects: 100% (55/55), done.[K
    remote: Total 66 (delta 7), reused 66 (delta 7), pack-reused 0[K
    Unpacking objects: 100% (66/66), done.



```python
%cd CV_Vinbigdata/
```

    /content/CV_Vinbigdata


# Install library


```python
!pip install dominate
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting dominate
      Downloading dominate-2.7.0-py2.py3-none-any.whl (29 kB)
    Installing collected packages: dominate
    Successfully installed dominate-2.7.0


# Dowload model

### Nếu lỗi khi bạn chạy trên colab là do section chưa load xong, check đường dẫn folder "/content/CV_Vinbigdata/checkpoints/colon2depth_512p" và chạy lại


```python
!gdown 1-jk4ddBheo1X3t12vKCPBxxubaOUO8S_  -O /content/CV_Vinbigdata/checkpoints/colon2depth_512p/80_net_D.pth
!gdown 13HXlby3-u9JQilWNnPWUSLnLJsoRNCxH  -O /content/CV_Vinbigdata/checkpoints/colon2depth_512p/80_net_G.pth
```

    Downloading...
    From: https://drive.google.com/uc?id=1-jk4ddBheo1X3t12vKCPBxxubaOUO8S_
    To: /content/CV_Vinbigdata/checkpoints/colon2depth_512p/80_net_D.pth
    100% 22.1M/22.1M [00:00<00:00, 160MB/s]
    Downloading...
    From: https://drive.google.com/uc?id=13HXlby3-u9JQilWNnPWUSLnLJsoRNCxH
    To: /content/CV_Vinbigdata/checkpoints/colon2depth_512p/80_net_G.pth
    100% 730M/730M [00:10<00:00, 71.7MB/s]


# Run test



### Data sample in folder "/content/CV_Vinbigdata/data_sample"
### Predict result in folder "/content/CV_Vinbigdata/data_sample/data_sample"


```python
!python ustc_test.py --name colon2depth_512p --no_instance --label_nc 0 --which_epoch 80
```

    ------------ Options -------------
    aspect_ratio: 1.0
    batchSize: 1
    checkpoints_dir: /content/CV_Vinbigdata/checkpoints
    cluster_path: features_clustered_010.npy
    data_type: 32
    dataroot: /content/drive/MyDrive/CV_depth_colonoscopy_1/datasets/colon2depth/
    display_winsize: 512
    engine: None
    export_onnx: None
    feat_num: 3
    fineSize: 256
    fp16: False
    gpu_ids: [0]
    how_many: 50
    input_nc: 3
    instance_feat: False
    isTrain: False
    label_feat: False
    label_nc: 0
    loadSize: 512
    load_features: False
    local_rank: 0
    max_dataset_size: inf
    model: pix2pixHD
    nThreads: 2
    n_blocks_global: 9
    n_blocks_local: 3
    n_clusters: 10
    n_downsample_E: 4
    n_downsample_global: 4
    n_local_enhancers: 1
    name: colon2depth_512p
    nef: 16
    netG: global
    ngf: 64
    niter_fix_global: 0
    no_flip: False
    no_instance: True
    norm: instance
    ntest: inf
    onnx: None
    output_nc: 1
    phase: test
    resize_or_crop: scale_width
    results_dir: ./results/
    serial_batches: False
    tf_log: False
    use_dropout: False
    use_encoded_image: False
    verbose: False
    which_epoch: 80
    -------------- End ----------------
    loading model...
    GlobalGenerator(
      (model): Sequential(
        (0): ReflectionPad2d((3, 3, 3, 3))
        (1): Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1))
        (2): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace=True)
        (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (5): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (6): ReLU(inplace=True)
        (7): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (8): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (9): ReLU(inplace=True)
        (10): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (11): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (12): ReLU(inplace=True)
        (13): Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (14): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (15): ReLU(inplace=True)
        (16): ResnetBlock(
          (conv_block): Sequential(
            (0): ReflectionPad2d((1, 1, 1, 1))
            (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
            (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (3): ReLU(inplace=True)
            (4): ReflectionPad2d((1, 1, 1, 1))
            (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
            (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          )
        )
        (17): ResnetBlock(
          (conv_block): Sequential(
            (0): ReflectionPad2d((1, 1, 1, 1))
            (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
            (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (3): ReLU(inplace=True)
            (4): ReflectionPad2d((1, 1, 1, 1))
            (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
            (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          )
        )
        (18): ResnetBlock(
          (conv_block): Sequential(
            (0): ReflectionPad2d((1, 1, 1, 1))
            (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
            (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (3): ReLU(inplace=True)
            (4): ReflectionPad2d((1, 1, 1, 1))
            (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
            (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          )
        )
        (19): ResnetBlock(
          (conv_block): Sequential(
            (0): ReflectionPad2d((1, 1, 1, 1))
            (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
            (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (3): ReLU(inplace=True)
            (4): ReflectionPad2d((1, 1, 1, 1))
            (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
            (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          )
        )
        (20): ResnetBlock(
          (conv_block): Sequential(
            (0): ReflectionPad2d((1, 1, 1, 1))
            (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
            (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (3): ReLU(inplace=True)
            (4): ReflectionPad2d((1, 1, 1, 1))
            (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
            (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          )
        )
        (21): ResnetBlock(
          (conv_block): Sequential(
            (0): ReflectionPad2d((1, 1, 1, 1))
            (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
            (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (3): ReLU(inplace=True)
            (4): ReflectionPad2d((1, 1, 1, 1))
            (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
            (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          )
        )
        (22): ResnetBlock(
          (conv_block): Sequential(
            (0): ReflectionPad2d((1, 1, 1, 1))
            (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
            (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (3): ReLU(inplace=True)
            (4): ReflectionPad2d((1, 1, 1, 1))
            (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
            (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          )
        )
        (23): ResnetBlock(
          (conv_block): Sequential(
            (0): ReflectionPad2d((1, 1, 1, 1))
            (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
            (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (3): ReLU(inplace=True)
            (4): ReflectionPad2d((1, 1, 1, 1))
            (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
            (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          )
        )
        (24): ResnetBlock(
          (conv_block): Sequential(
            (0): ReflectionPad2d((1, 1, 1, 1))
            (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
            (2): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (3): ReLU(inplace=True)
            (4): ReflectionPad2d((1, 1, 1, 1))
            (5): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1))
            (6): InstanceNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          )
        )
        (25): Sequential(
          (0): Upsample(scale_factor=2.0, mode=nearest)
          (1): ReflectionPad2d((1, 1, 1, 1))
          (2): Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1))
        )
        (26): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (27): ReLU(inplace=True)
        (28): Sequential(
          (0): Upsample(scale_factor=2.0, mode=nearest)
          (1): ReflectionPad2d((1, 1, 1, 1))
          (2): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1))
        )
        (29): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (30): ReLU(inplace=True)
        (31): Sequential(
          (0): Upsample(scale_factor=2.0, mode=nearest)
          (1): ReflectionPad2d((1, 1, 1, 1))
          (2): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1))
        )
        (32): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (33): ReLU(inplace=True)
        (34): Sequential(
          (0): Upsample(scale_factor=2.0, mode=nearest)
          (1): ReflectionPad2d((1, 1, 1, 1))
          (2): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1))
        )
        (35): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (36): ReLU(inplace=True)
        (37): ReflectionPad2d((3, 3, 3, 3))
        (38): Conv2d(64, 1, kernel_size=(7, 7), stride=(1, 1))
        (39): Tanh()
      )
    )
    loading data
    inferencing
    /content/CV_Vinbigdata/models/pix2pixHD_model.py:128: UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
      input_label = Variable(input_label, volatile=infer)
    process image... data_sample/T1_L1_3_resized_FrameBuffer_0189.png
    process image... data_sample/T1_L1_2_resized_FrameBuffer_0325.png
    process image... data_sample/T1_L1_2_resized_FrameBuffer_0236.png
    process image... data_sample/T1_L1_1_resized_FrameBuffer_0203.png


# show result


```python
import glob
import cv2
import matplotlib.pyplot as plt
data_test = glob.glob('/content/CV_Vinbigdata/data_sample/*.png')
len(data_test)
```


```python

for i in data_test:
  predict_path = i.replace("data_sample","data_sample/data_sample")
  img = cv2.imread(i)
  predict_img = cv2.imread(predict_path)
  plt.figure(figsize=(12, 12))
  plt.subplot(1, 2, 1)
  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
  plt.subplot(1,2,2)
  plt.imshow(cv2.cvtColor(predict_img, cv2.COLOR_BGR2RGB))
  plt.show()
```


    
![png](static/static1.png)
    



    
![png](static/static2.png)
    



    
![png](static/static3.png)
    



    
![png](static/static4.png)
    



## Acknowledgments
This code borrows from [Self-Supervised-Depth-Estimation-for-Colonoscopy](https://github.com/ckLibra/Self-Supervised-Depth-Estimation-for-Colonoscopy.git).


## REFERENCE
[1] Cheng, K., Ma, Y., Sun, B., Li, Y., & Chen, X. (2021). Depth Estimation for Colonoscopy Images with Self-supervised Learning from Videos. Medical Image Computing And Computer Assisted Intervention – MICCAI 2021, 119-128. doi: 10.1007/978-3-030-87231-1_12.

[2] Shao, S., Pei, Z., Chen, W., Zhu, W., Wu, X., Sun, D., & Zhang, B. (2021). Self-Supervised Monocular Depth and Ego-Motion Estimation in Endoscopy: Appearance Flow to the Rescue. Retrieved 28 September 2022, from https://arxiv.org/abs/2112.08122.

[3] Liu, X., Sinha, A., Ishii, M., Hager, G., Reiter, A., Taylor, R., & Unberath, M. (2019). Dense Depth Estimation in Monocular Endoscopy with Self-supervised Learning Methods. Retrieved 28 September 2022, from https://arxiv.org/abs/1902.07766.

[4] Patel, Dhara & Upadhyay, Saurabh. (2013). Optical Flow Measurement using Lucas Kanade Method. International Journal of Computer Applications. 61. 6-10. 10.5120/9962-4611.

[5] Sun, D., Yang, X., Liu, M., & Kautz, J. (2017). PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume. Retrieved 28 September 2022, from https://arxiv.org/abs/1709.02371

[6] Rau, A., Edwards, P.E., Ahmad, O.F., Riordan, P., Janatka, M., Lovat, L.B., Stoyanov, D.: Implicit domain adaptation with conditional generative adversarial networks for depth prediction in endoscopy. Int. J. Comput. Assist. Radiol. Surg. 14(7), 1167–1176 (2019). https://doi.org/10.1007/s11548-019-01962-w