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

Tricks to use:
- label smoothing
- check "Bag of tricks to ..."
- what if we train class attributes (but initialize from ready ones, ofc)?

TODO:
- B in RGB has wrong mean for our images (we should use other mean instead of imagenet-like mean)
- We have quite large logits
- Our model overfits
- In ZSL-setting forgetting measure is not that correct since we can have better accuracy for the task BEFORE we have seen it. We should compute `max` only since the task when we have first encountered this task. Besides, it can be negative for ZSL, which is strange.
- For memory replay, try taking only those samples for each classifier is very certain. This potentially will allow to take very good images. Actually, we can verify (manually or automatically) if such images are really better.
- Perceptual loss instead of MSE loss?
- New metric for LLL: plasticity, which is how much tasks can an agent learn. Because some regularizations can be good, but they are too constraining for acquiring new tasks.
