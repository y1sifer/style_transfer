from django.shortcuts import render, get_object_or_404
from django.views import generic
from .models import Content_image, Style_image
from django.utils import timezone


def index(request):
    if request.method == "POST":
        content_img = Content_image(
            img = request.FILES.get('content_img'),
            name = request.FILES.get('content_img').name,
            pub_time = timezone.now()
        )
        style_img = Style_image(
            img=request.FILES.get('style_img'),
            name=request.FILES.get('style_img').name,
            pub_time=timezone.now()
        )
        content_img.save()
        style_img.save()
        content = {
            'content_img':content_img,
            'style_img':style_img
        }
        return render(request, "style_transfer/index.html", content)
    if Content_image.objects.order_by('-pub_time').count() == 0:
        return render(request, 'style_transfer/index.html')
    content_img = Content_image.objects.order_by('-pub_time')[-1]
    style_img = Style_image.objects.order_by('-pub_time')[-1]
    # content_img = get_object_or_404(Content_image)
    # style_img = get_object_or_404(Style_image)
    content = {
        'content_img':content_img,
        'style_img':style_img
    }
    return render(request, 'style_transfer/index.html', content)



def showImg(request):
    content_img = Content_image.objects.order_by("-pub_time")[0]
    style_img = Style_image.objects.order_by("-pub_time")[0]
    content = {
        'content_img':content_img,
        'style_img':style_img
    }
    return render(request, "style_transfer/showing.html", content)