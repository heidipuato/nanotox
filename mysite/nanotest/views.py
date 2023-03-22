from django.shortcuts import render

# Create your views here.
def home(request):
    return render(request,"nanotest/home.html")

def test(request):
    return render(request,"nanotest/test.html")
