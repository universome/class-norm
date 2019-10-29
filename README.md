Our setup:
- Our main model is discriminator, which performs classification. The setup is very similar to CIZSL paper in that sense.
- Next, we have an additional A-GEM loss, which makes discriminator not to forget older examples.

Ideas:
- We can use class prototypes as task descriptors. To get a prototype we can just compute average VGG feature. Of course, this will be cheating, but will be a good PoC.
- Like in editable neural networks, but we can make outputs to be unchanged for just some random inputs.
- Can we change episodic memory in the following way. We train a classification model together with a generative one. And instead of storing examples, we just generate them. We can generate them with the corresponding class labels. And we can do it without class labels by the following way. On each iteration we generate a batch of examples, remember our model's outputs on them. And then perform an optimization step in such a way that outputs for a generated batch does not change (either by regularization or by projecting the gradient).
- Is it really better to use episodic memory for projecting the gradient? Maybe it will be more beneficial just to mix examples from the episodic memory?
- It will be an interseting survey to see how diffferent LLL methods work on tasks which come from different data distributions. I.e. if we have completely different tasks.
- Are adversarially robust classifiers less prone to catastrophic forgetting?
- Recent LTH works ([1](https://arxiv.org/abs/1905.07785), [2](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8852405&tag=1)) imply that winning tickets are quite the same for similar tasks. In our LLL setup we use very similar tasks (subsets of the same dataset). This means that EWC/MAS/similar methods shouldn't work well, because they restrict changing the winning ticket, and those weights which these methods allow to change — we do not need to, because they are not a part of the winning ticket.
- What if we'll hallucinate future classes?
- Can we have a scheduled learning, i.e. a task sequence, that is made up in such a way that it's easier for us to learn? Just like in school
- We can generalize AUSUC to multi-task settings. Imagine we have not just Seen/Unseen groups, but Task 1, Task 2, ..., Task N groups. For each group we have a specified threshold t_i that we vary from -\infty to +\infty. As a result we get the surface in N-dimensional space, that should reflect the interconnection between all the tasks. The main problem is that we cannot compute it. And maybe this surface is diverging: imagine 3 tasks with 2 thresholds t_2 and t_3 to control the influence of Task 2 and Task 3. Then for t_3 = -\infty we have a normal AUSUC between Task 1 and Task 2. 

Tricks to use:
- label smoothing
- check "Bag of tricks to ..."
- what if we train class attributes (but initialize from ready ones, ofc)?
- penalize other logits sometimes to make model be prepared for operating in "distinguish between all-classes" regime. For A-GEM we could use episodic memory for this.

TODO:
- B in RGB has wrong mean for our images (we should use other mean instead of imagenet-like mean)
- We have quite large logits
- Our model overfits
- In ZSL-setting forgetting measure is not that correct since we can have better accuracy for the task BEFORE we have seen it. We should compute `max` only since the task when we have first encountered this task. Besides, it can be negative for ZSL, which is strange.
- For memory replay, try taking only those samples for each classifier is very certain. This potentially will allow to take very good images. Actually, we can verify (manually or automatically) if such images are really better.
- Perceptual loss instead of MSE loss?
- New metric for LLL: plasticity, which is how much tasks can an agent learn. Because some regularizations can be good, but they are too constraining for acquiring new tasks.
- Will it be better if we sample random classes from Generator on each trianing step? And not classes from the current batch?
