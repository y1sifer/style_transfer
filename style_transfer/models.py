from django.db import models


# 加载图片
class Content_image(models.Model):
    img = models.ImageField(upload_to='content_img')
    name = models.CharField(max_length=20)
    pub_time = models.DateTimeField("data published")

class Style_image(models.Model):
    img = models.ImageField(upload_to='style_img')
    name = models.CharField(max_length=20)
    pub_time = models.DateTimeField("data published")