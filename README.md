## My Leukemia Detection Project 

Video Breakdown: https://youtu.be/vXWVIPjk69c
(Watch in 1.5x)

Each file will have a brief breakdown of what it does, and its use! These are only the final products, but if you have any other questions on how I came to certain decisions, please ask! My LinkedIn & email are in my bio. 

In this project, we create a CNN to detect Leukemia based on cleaned cell X-rays, use GRadcam++ to better explain model predictions, and then run a Streamlit app with a local LLM agent made to be asked questions and assist physicians with clinical decisions. The model and agent can both be greatly improved, but the point of this is more a proof of concept (to show employers). I have future plans to change the model to train on negative cells, then predict based on the variance from these cells. I found a previous study that had an F1-score in the 90s, which sounds more impressive than it is. It was calculated based on both training and validation data, which is an underhanded way to make your model seem better. 

The data is from a 2019 competition found below: 

Gupta, A., & Gupta, R. (2019). ALL Challenge dataset of ISBI 2019 [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/tcia.2019.dc64i46r

The naming scheme is as such (directly from the competition overview):

"""
Cancer cell images' naming convention: UID_P_N_C_all
● UID_P -> where P=1,2.... signifies the subject ID.
● UID_P_N: where N=1,2,3... represent the image number
● UID_P_N_C: where C=1,2,3... represents the cell count. (More than one cell can be found in a particular microscopic image)
● UID_P_N_C_all: The ‘all’ tag represent the class to which the cell belongs, in this case, ‘ALL’ or cancer class.

Similarly, the naming convention for normal (healthy) cell images is as follows: 
UID_HS_N_C_hem, where H denotes healthy/normal subject, S denotes the healthy subject's ID, N denotes the image number, C denotes the cell count, and hem tag, in the end, denotes the normal subjects' cell.
"""

Here is an example of the cell image X-rays. 

<img width="794" height="905" alt="image" src="https://github.com/user-attachments/assets/f511781c-a3f4-42ea-a9f7-04c1279d4335" />
<img width="794" height="905" alt="image" src="https://github.com/user-attachments/assets/2b545b32-8879-427f-a70f-07c832736d48" />

The training set consisted of 7272 leukemia patient cell X-rays and 3389 non-leukemia patient cell X-rays. The validation dataset consisted of 1867 cell images. 

# Model Creation

This file is a qmd and has more in-depth notes on each step.

This file loads in all data from the Kaggle API, cleans it, creates a CNN, and predicts based on an individual cell level. We end up with an AUC of .7082 on the validation data and an accuracy of .7172. This is on a by cell level. In the demo, we predict Leukemia based on the average cell. This could be better optimized, but an average is the simplest way to do this, and the point of this project is a demo, not to stop cancer!

<img width="1001" height="547" alt="AUC Graph" src="https://github.com/user-attachments/assets/4e6efbeb-2d2b-471f-98fe-7173ac58da60" />

<img width="560" height="455" alt="image" src="https://github.com/user-attachments/assets/7030e887-bd1b-4ae1-abfc-b6c693f0332c" />

# Computer Vision Analysis

This file is a qmd and has more in-depth notes on each step.

Here, we use Gradcam++ to show where on each cell the model sees Leukemia. This is so cool! Neural Nets are black boxes, so being able to explain the model's prediction can be very difficult. Getting to see where the model thinks Leukemia is helps break down its decisions. It alos is also very helpful for clinicians to better understand what the model is seeing they could be missing! 

Here is an example:
<img width="790" height="2312" alt="image" src="https://github.com/user-attachments/assets/0b910927-68a7-4b27-bf89-9e83feab3799" />

# Agent

This file gives context and rules to the LLM agent in order to best complette the tasks we want it to. It gives it prompts, to understand the context of the conversation, and sample code pieces to answer questions. This allows the model to give data-backed decisions, and let clinicians vibe code with out typign any Python! It is also entirely secure, as it is a local model (phi3:mini). In the future we would use a better model, but I want to be able to run this on my 8 GB RAM laptop while recording it for a demo. 

This also originally had a .md file loaded in, which did not help. Eventually, all the code was added to allow it to work better! 

# App

Finally, we have the streamlit, which is the front-end UI for clinicians. The agent can still be kinda finnicky, so I woudl suggest saying which patient I ayou are asking about. I would also suggest using the drop-down on the left, selecting which ID as well. It is pretty cool to just say you want analysis and have it give you the yes/no for Leukemia, then give the heatmap. I would also strongly recommend using the datasets of one individual (2 and 86). 

