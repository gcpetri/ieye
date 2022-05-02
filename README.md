# Final Project Report

## Gregory Petri ##
CSCE 489 <br>
5/2/2022

# Summary

The project consists of a frontend web server that uses a device's camera to take photos of a user's eye. The user then has the option to "signup" or "login". The the user chooses to "signup", the backend performs feature detection with <code>python-opencv SIFT detection</code> on their eye image to attain metrics and store them in a database with the user's nickname as the key. SIFT detection intelligently and quickly discovers keypoints and calculates the descriptors around these keypoints. If the user chooses "login", the backend still performs the same feature detection process on their eye image, but does not save the data to the database. Instead, the backend fetches every user's data from the database finds all the matches between that data and the "logging-in" user's keypoints and descriptors. With <code>python-opencv FLANN matching</code>, SIFT descriptors can be easily compared and the matches plotted in a graphic. Once the matching is complete, the user is then identified with the nickname corresponding to the maximum matches.

# Instructions

**Open the web app**

_Locally_

```bash
.\init-ps1 # PowerShell
# - or -
./init.sh # Bash
```

<a href="http://localhost:5000/" target="_blank">http://localhost:5000/</a>

_Online_

<a href="https://ieye.herokuapp.com/" target="_blank">https://ieye.herokuapp.com/</a>

**View the Home Page**

![home](https://github.com/gcpetri/ieye/blob/master/Captures/home.PNG?raw=true)

**Take an Eye Picture**

_And enter a unique nickname*_

![home-eye](https://github.com/gcpetri/ieye/blob/master/Captures/home-eye.PNG?raw=true)

**Sign Up**

_Original Image_

![signup-1](https://github.com/gcpetri/ieye/blob/master/Captures/signup-1.PNG?raw=true)

_Resized, Smoothed, and Contrasted Image_

![signup-2](https://github.com/gcpetri/ieye/blob/master/Captures/signup-2.PNG?raw=true)

_SIFT KeyPoints and Descriptors Display_

![signup-3](https://github.com/gcpetri/ieye/blob/master/Captures/signup-3.PNG?raw=true)

**Back to Home and Take Another Image**

![home-login](https://github.com/gcpetri/ieye/blob/master/Captures/home-login.PNG?raw=true)

**Login**

![login](https://github.com/gcpetri/ieye/blob/master/Captures/login.PNG?raw=true)

# Discussion

**Implementation**

I started the implementation with bare-bones code. I created a custom gaussian kernel, harris detector, and heuristic for the keypoints. Some of the code for these is still in <code>harris.py</code>. Unfortunately, not only did my implementations give questionably accurate and slow results, but also calculating keypoint descriptors proved too challenging. In the interest of run-time and my busy schedule, I turned to the miracle that is the <code>python-opencv</code> library. This allowed me to quickly compute descriptors and make them orientation invariant with <code>cv2.SIFT_create().detectAndCompute()</code>. This saved me a lot of time, especially since it comes with a compatable matching function <code>cv2.FlannBasedMatcher()</code>. The main challenge was storing the images, keypoints, and descriptors in a database. I decided to use MongoDB Atlas because it is free and JSON objects can be large. Once the I solved converting numpy arrays to binary and vice-versa, the remaining work was trivial.

# Conclusion

I worked really hard on this so I hope it is enjoyed!

The closer the perimeter of the iris is to matching the circumference of the white circle, the better the results will be. To see the intended results, go to <a href="https://ieye.herokuapp.com/test" target="_blank">https://ieye.herokuapp.com/test</a>

![test](https://github.com/gcpetri/ieye/blob/master/Captures/test.PNG?raw=true)
