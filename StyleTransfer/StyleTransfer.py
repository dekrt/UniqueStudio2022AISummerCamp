# %%
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import copy

# %%
class GramMatrix(nn.Module):
    def forward(self, input):
        """
        b for batch size, n for number of feature maps,
        h, w for height, width
        """
        b, n, h, w = input.size()
        features = input.view(b * n, h * w)
        gm = torch.mm(features, features.t())
        return gm.div(b * n * h * w)

# %% [markdown]
# 定义内容损失

# %%
class ContentLoss(nn.Module):
    def __init__(self, weight, target):
        super(ContentLoss, self).__init__()
        self.weight = weight
        self.target = target.detach() * weight
        self.loss_fn = nn.MSELoss()

    def forward(self, input):
        self.loss = self.loss_fn(input * self.weight, self.target)
        self.output = input
        return self.output

    def backward(self):
        self.loss.backward(retain_graph=True)
        return self.loss

# %% [markdown]
# 定义风格损失

# %%
class StyleLoss(nn.Module):
    def __init__(self, weight, target):
        super(StyleLoss, self).__init__()
        self.weight = weight
        self.target = target.detach() * weight
        self.loss_fn = nn.MSELoss()
        self.gram = GramMatrix()

    def forward(self, input):
        self.output = input.clone()
        self.GM = self.gram(input)
        self.GM.mul_(self.weight)
        self.loss = self.loss_fn(self.GM, self.target)
        return self.output

    def backward(self):
        self.loss.backward(retain_graph=True)
        return self.loss

# %% [markdown]
# 加载、展示图片

# %%
def load_img(path):
    transformer = transforms.Compose([transforms.Resize([256, 256]),
                                     transforms.ToTensor()])
    img = Image.open(path)
    img = transformer(img)
    img = img.unsqueeze(0)
    return img


# %%
content_img = load_img("./content.jpg")
style_img = load_img("./style.jpg")
content_img = Variable(content_img)
style_img = Variable(style_img)
if torch.cuda.is_available():
    content_img = content_img.cuda()
    style_img = style_img.cuda()

# %%
def show_img(img, title=None):
    unloaded = transforms.ToPILImage()
    img = img.clone().cpu()
    img = img.squeeze(0)
    img = unloaded(img)
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

# %%
plt.figure()
show_img(style_img, title='Style Image')
show_img(content_img, title='Content Image')

# %% [markdown]
# 加载预训练的VGG16模型

# %%
vgg = models.vgg16(pretrained=True).features
my_model = nn.Sequential()
if torch.cuda.is_available():
    vgg = vgg.cuda()
    my_model = my_model.cuda()
model = copy.deepcopy(vgg)

# %%
print(vgg)

# %% [markdown]
# 传入自定义的损失函数

# %%
content_layers = ["Conv_5", "Conv_6"]
style_layers = ["Conv_1", "Conv_2", "Conv_3", "Conv_4", "Conv_5"]
content_losses = []
style_losses = []
content_weight = 1
style_weight = 1000
gram = GramMatrix()
i = 1  # index

if torch.cuda.is_available():
    gram = gram.cuda()
    my_model = my_model.cuda()

for layer in list(model):

    if isinstance(layer, nn.Conv2d):
        name = 'Conv_' + str(i)
        my_model.add_module(name, layer)
        if name in content_layers:
            target = my_model(content_img).clone()
            content_loss = ContentLoss(content_weight, target)
            my_model.add_module("content_loss_"+str(i), content_loss)
            content_losses.append(content_loss)
        if name in style_layers:
            target = my_model(style_img).clone()
            target = gram(target)
            style_loss = StyleLoss(style_weight, target)
            my_model.add_module("style_loss_"+str(i), style_loss)
            style_losses.append(style_loss)

    if isinstance(layer, nn.ReLU):
        name = "Relu_"+str(i)
        my_model.add_module(name, layer)
        i += 1

    if isinstance(layer, nn.MaxPool2d):
        name = "MaxPool_"+str(i)
        my_model.add_module(name, layer)
print(my_model)

# %%
input_img = content_img.clone()

parameter = nn.Parameter(input_img.data)
optimizer = torch.optim.LBFGS([parameter])

# %% [markdown]
# 进行训练

# %%
epoch = 500
n = [0]
while n[0] <= epoch:
    def closure():
        optimizer.zero_grad()
        style_score = 0
        content_score = 0
        parameter.data.clamp_(0, 1)
        my_model(parameter)
        for sl in style_losses:
            style_score += sl.backward()

        for cl in content_losses:
            content_score += cl.backward()

        n[0] += 1
        if n[0] % 50 == 0:
            print('{} Style Loss : {:4f} Content Loss: {:4f}'.format(n[0], style_score.item(), content_score.item()))

        return style_score + content_score
    optimizer.step(closure)

# %%
parameter.data.clamp_(0,1)
plt.figure()
show_img(parameter.data, title="Output Image")

# %%



