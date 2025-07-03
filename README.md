# Sel supervised project (SSL_project)

This project aims to explore self-supervised learning. The theory behind is writen on a [google docs](https://docs.google.com/document/d/1A8WWZ6pXrtEehzWBwBQnVNTihtZqCWcSWt40w7mc4SU/edit?tab=t.0)

## Step up process:

Install requirements file using 
 ```pip install -r requirement.txt```

If you have a GPU, install torch with cuda, follow the [link](https://pytorch.org/get-started/locally/). I use cuda 12.8 so used the command ```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128```.

You will need to check the Cuda version. Go to command line and type "nvidia-smi". This Stack overflow [link](https://stackoverflow.com/questions/9727688/how-to-get-the-cuda-version), show how to do so. 

## Useful Extension 

* Black
* Isort
* Flake8

These for reformatting the code to meet coding standard and make code easier to read.

Follow steps on [link](https://dev.to/facepalm/how-to-set-formatting-and-linting-on-vscode-for-python-using-black-formatter-and-flake8-extensions-322o#:~:text=You%20can%20set%20max%20length,Flake8%20Options)

## Simiplified Pipeline
```
img = get_image()
view1 = augment(img)
view2 = augment(img)

embedding1 = model(view1)
embedding2 = model(view2)

loss = contrastive_loss(embedding1, embedding2, negatives)
```

### In pros
* Two augmented views of the same image (positive pair)
* Many other images as negatives
* A base encoder (e.g., ResNet18)
* A projection head (MLP that maps embeddings to contrastive space)
* A contrastive loss function (e.g., NT-Xent)