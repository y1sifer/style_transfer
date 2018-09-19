from django.shortcuts import render, get_object_or_404
from django.views import generic
from .models import Content_image, Style_image
from django.utils import timezone
from .model.imple_transfer import StyleTransfer

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
    style_transfer = StyleTransfer('.'+content_img.img.url,'.'+style_img.img.url)
    style_transfer.load_model()
    img1 = style_transfer.transformed_image()
    style_transfer.train()
    img2 = style_transfer.transformed_image()
    # path = '/Users/2black/2black_workspace/django_test1/mysite/style_transfer/static/style_transfer/images/result.jpg'
    # style_transfer.save_image(img, '/Users/2black/2black_workspace/django_test1/mysite/style_transfer/static/style_transfer/images/result.jpg')
    path = '/home/egg/2black_workspace/style_transfer/media/'
    style_transfer.save_image(img1, path+'result1.jpg')
    style_transfer.save_image(img2, path+'result2.jpg')
    content = {
        'content_img':content_img,
        'style_img':style_img,
        'result1_img_url': "/media/result1.jpg",
        'result2_img_url': "/media/result2.jpg"

    }
    Content_image.objects.all().delete()
    Style_image.objects.all().delete()
    return render(request, "style_transfer/showing.html", content)