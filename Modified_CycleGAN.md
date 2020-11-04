# Modified CycleGAN

* [Enhancing the Accuracies of Age Estimation With Heterogeneous Databases Using Modified CycleGAN](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8894442)

## Implementation

* Tensorflow >= 2.0
* python 3.5.0
* Ubuntu 18.04

## Train(FLAGS)

* "A_txt_path": A training text in "A to B" direction

  * text: image name list

    * | A_image.txt                                      |
      | ------------------------------------------------ |
      | image1.jpg<br/>image2.jpg<br/>image3.jpg<br/>... |

* "A_img_path": A training image path (Must include "/" in the last of sentence)
* "n_A_images": Number of training A set
* "B_txt_path": B training  text in "B to A" direction
* "B_img_path": B training image path (Must include "/" in the last of sentence)
* "n_B_images": Number of training B set
* "train": Set "True" in bool type
* "pre_checkpoint": Set "False" in bool type (but if you want to continue learning, Set "True")
* "pre_checkpoint_path":  Path of weight files to load (In train from scratch, you can leave it blank)
* "save_checkpoint": Path where weights will be saved
* "sample_dir": Path where training samples will be saved

## Test(FLAGS)

* "A_test_img": A testing image path (Must include "/" in the last of sentence)
* "A_test_txt": A testing  text
* "A_n_images": Number of testing A set
* "B_test_img": B testing image path (Must include "/" in the last of sentence)
* "B_test_txt": B testing  text
* "B_n_images": Number of testing B set
* "test_dir": "A2B" or "B2A"
* "A_test_output": if FLAGS.test_dir is "A2B"
* "B_test_output": if FLAGS.test_dir if "B2A"

 ## Result

* A to B

![image-20201104131942527](C:\Users\Yuhwan\AppData\Roaming\Typora\typora-user-images\image-20201104131942527.png)

* B to A

![image-20201104132050941](C:\Users\Yuhwan\AppData\Roaming\Typora\typora-user-images\image-20201104132050941.png)