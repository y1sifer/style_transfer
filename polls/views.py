from django.shortcuts import render, get_object_or_404

# Create your views here.
from django.http import HttpResponse, HttpResponseRedirect
from .models import  Question, Choice
from django.template import loader
from django.http import Http404
from django.urls import reverse
from django.views import generic
from django.utils import timezone
import datetime



def index(request):
    # latest_question_list = Question.objects.order_by('-pub_date')
    # output = ', '.join([q.question_text for q in latest_question_list])
    # return HttpResponse(output)

    # latest_question_list = Question.objects.order_by('question_text')
    # template = loader.get_template('polls/index.html')
    # context = {
    #     'latest_question_list': latest_question_list,
    # }
    # return HttpResponse(template.render(context, request))
    latest_question_list = Question.objects.order_by('-pub_date')
    context = {
        'latest_question_list' : latest_question_list,
    }
    return render(request, 'polls/index.html', context)

def detail(request, question_id):
    # return HttpResponse('You\'re looking at question {}.'.format(question_id))

    # try:
    #     question = Question.objects.get(pk=question_id)
    # except Question.DoesNotExist:
    #     raise Http404('Question does not exist')

    question =  get_object_or_404(Question, pk=question_id)
    return render(request, 'polls/detail.html', {'question': question})
def results(request, question_id):
    # response = 'You\'re looking at the results of question %s.'
    # return HttpResponse(response % question_id)

    question = get_object_or_404(Question, question_id)
    return render(request, 'polls/result.html', {'question': question})

def vote(request, question_id):
    # return HttpResponse('You\'re voting on question'.format(question_id))
    question = get_object_or_404(Question, pk=question_id)
    try:
        selected_choice = question.choice_set.get(pk=request.POST['choice'])
    except(KeyError, Choice.DoesNotExist):
        return render(request, 'polls/detail.html', {
            'question': question,
            'erroe_message': 'You didn\'s select a choice.',
        })
    else:
        selected_choice.votes += 1
        selected_choice.save()

        return HttpResponseRedirect(reverse('poll:results', args=(question_id,)))



class IndexView(generic.ListView):
    template_name = 'polls/index.html'
    context_object_name = 'latest_question_list'

    def get_queryset(self):
        return Question.objects.filter(pub_date__lte=timezone.now()).all()

class DetailView(generic.DetailView):
    model = Question
    template_name = 'polls/detail.html'

    def get_queryset(self):
        return Question.objects.filter(pub_date__lte=timezone.now())



class ResultsView(generic.DetailView):
    model = Question
    template_name = 'polls/result.html'




































