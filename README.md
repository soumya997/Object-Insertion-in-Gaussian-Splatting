# Object Insertion in Gaussian Splatting

<img src="media/garden_canvas_dslr_360_clip1.gif">

### Features:

- Applied mask-guided 3DGS training for background-extracted models.
- Utilized Floater-Free MCMC with Bilateral Grid training for large scenes.
- Employed KNN-based outlier removal for both points and gaussians.
- Adjusted 3DGS models to ensure accurate placement within large scenes.
- Reduced COLMAP reconstruction time by 10x by extracting features from 512×512 images and upscaling keypoints for 1024×1024 undistortion, enabling fast SFM on low-res images while training with high-res data. [**Work in Progress**]




### Files:

1. `src/insert_canvas_in_garden.py` : insert canvas ply to garden ply
2. `src/colmap_rerun.py` : Used for Rerun logging 
3. `with_undistort_colmap.py`: do reconstruction on resized image, then undistort for larger image. The large image can be used for high fedality 3dgs reconstruction.
4. `colmap_hloc.py`: normal `colmap+hloc` reconstruction. 

---


### Sample Dataset
Download the dataset from here:
- [**garden**](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip)
- [**Canvas**](https://www.dropbox.com/scl/fi/2k5xfxpani744dzbxco5h/obj_insert_canvas_data_blog.zip?rlkey=pxwui6w4h7s8ql3eqpxzq51ls&st=8fl2f5f1&dl=1)




### Training Large Scene Splat using Gsplat

MCMC + Bilateral grid has an ability to create floater free gaussian models.
So, we will be using for that purpose.

You can limit the number of gaussians also, which helps in reducing the file size.

```
$ CUDA_VISIBLE_DEVICES=0 python examples/simple_trainer.py mcmc \
 --use_bilateral_grid \
 --data_dir /path/of/garden/folder \
 --result_dir /path/to/result/garden \
 --data_factor 1
```


- --data_dir: This is the directory where the images and sparse folders are located.
- --result_dir: Specify an empty directory to store the results. I recommend creating a directory named results and, inside it, another folder named garden to organize the output.
- --data_factor: This refers to the resolution at which 3DGS processes your images. A value of 1 retains the original resolution.



### Training Object Splat

```
$ rembg p /path/to/object/images /path/to/object/images_rembg
```


Once you have removed the background, you will have RGBA images. To convert these images to .jpg format with a white background, run the command below:


```
$ cd /path/to/object/images_rembg
$ for file in *.png; do convert "$file" -background white -flatten "${file%.png}.jpg"
$ mv /path/to/object/images /path/to/object/images_org
$ mv /path/to/object/images_rembg /path/to/object/images
```


Train mask guided gaussian splatting using the inria code

```
$ python train.py -s /path/to/canvas/folder \
 -m /path/to/results/folder \
 --white_background
```


This command is similar to the gsplats command, but we are using three specific flags:

- -s: Equivalent to --data_dir.
- -m: Equivalent to --result_dir.
- --white_background: Ensures the background is removed from the splat of the object.

### Removing Outliers from 3D Gaussian Splatting

```
$ 3dgsconverter -i input_3dgs.ply -o output_cc.ply -f cc --remove_flyers
$ 3dgsconverter -i output_cc.ply -o output_3dgs.ply -f 3dgs
```


### Merge Object and Scene Splats


```
$ python insert_canvas_in_garden.py
```

This approach allows us to directly operate on the Gaussian .ply file instead of converting it to a point cloud, performing the operations, and then converting it back to a Gaussian .ply file. This eliminates unnecessary back-and-forth conversions.

The output .ply file can be easily be visualized using the supersplat visualizer.

