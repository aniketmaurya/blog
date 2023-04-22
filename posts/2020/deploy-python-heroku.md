---
title: Deploy Machine Learning Web Apps for Free
description: Learn to deploy Python Tensorflow & FastAPI Web app on Heroku Cloud in 5 minutes.
aliases:
- /python/2020/10/15/deploy-python-heroku
badges: true
categories:
- python
date: '2020-10-15'

image: https://cdn-images-1.medium.com/max/6706/0*m45LdHFZa0noxH-0
keywords: tensorflow, fastapi, python, machine learning, computer vision, deploy,
  cloud
layout: post
toc: false

---

Do you want to deploy your Python code to web for free? Read this post to know the step-by-step procedure to host your web app for free on Heroku cloud.

<!-- ![Photo by [Kevin Ku](https://unsplash.com/@ikukevk?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)](https://cdn-images-1.medium.com/max/6706/0*m45LdHFZa0noxH-0)*Photo by [Kevin Ku](https://unsplash.com/@ikukevk?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)* -->


Deploying a Machine Learning model is a difficult task due to the requirement of large memory and powerful computation. This tutorial focuses on a simple deployment technique that can be used to deploy any Python web app for free.

> Read my previous article to learn how to build an [“Image classification web app with FastAPI and Tensorflow”](https://blog.aniketmaurya.ml/python/2020/07/26/fastapi-tf-webapp.html)


If you prefer a video tutorial then click on the video below:

<center><iframe width="540" height="480" src="https://www.youtube.com/embed/9gSkdEWx_VA" frameborder="10" allowfullscreen></iframe></center>

# Step 1
First of all, you will need a [Heroku](http://heroku.com) id, so go now and register for a free account.

For deploying any Python app on Heroku, we need three files- requirements.txt, runtime.txt, and Procfile.

**requirements.txt** is a normal text file that contains Python packages required to run the app.

**runtime.txt** is a text file that will contain the Python Version you want your app to run on.

**Procfile** will contain the command to launch your web app. For example, you can use

    method 1
    `python application/server/main.py`

    or uvicorn if you are deploying a uvicorn server
    `uvicorn application.server.main:app`

# Step 2
Go to your Heroku [dashboard](https://dashboard.heroku.com/apps) then click on **New** and **create a new app**

>![](https://cdn-images-1.medium.com/max/5724/1*mXrC1C1oudHwF3KAwxUegg.png)

<br>
Enter your **App name** and select the Server region that is nearest to your location and click on **Create app**

>![](https://cdn-images-1.medium.com/max/2544/1*_r6QhIjusWh1D2NbzN_iwA.png)

<br>
# Step 3
After you create the app, you will see the deployment methods — Heroku Git, GitHub, and Container Registry. I will use the GitHub method. For this just push your code repository to your GitHub account and then connect to GitHub on Heroku.

![](https://cdn-images-1.medium.com/max/3704/1*Y6f9uWQ7Nf9qXEsG6Zv0pA.png)

Then search the repository and connect it to your Heroku app.

>![](https://cdn-images-1.medium.com/max/4836/1*YpVzIgY35QUqq4uS5FKcxw.png)

After this, you will see a deploy button, select the branch of your Git repository that you want to be deployed, and click on **deploy**.

Then Heroku will automatically start your deployment 🎉🎉
After deployment, you can access your app from any web browser 🔥


<br>
<hr>

Hope this article helped you in the deployment of your web app. If you have some feedback or suggestion please let me know in the comment section.

> Follow me on **Twitter**: https://twitter.com/aniketmaurya

> Subscribe to my **YouTube channel:** https://www.youtube.com/channel/UCRuFsj94hWecPkuEr4f5Xww
