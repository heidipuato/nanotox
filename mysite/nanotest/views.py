from django.shortcuts import render
from .models import *
from .forms import *
import joblib
import numpy as np

# Create your views here.
def home(request):
    return render(request,"nanotest/home.html")

def test(request):
    form = FeatureForm

    data = {"form":form}
    return render(request,"nanotest/test.html", data)

def about(request):
    return render(request,"nanotest/about.html")

def predict(request):
   cls=joblib.load('model.sav')
   lis=[]

   val01 = request.POST.get('val1')
   val1 = float(val01)
   val02 = request.POST.get('val2')
   val2 = float(val02)
   val03 = request.POST.get('val3')
   val3 = float(val03)
   val04 = request.POST.get('val4')
   val4 = float(val04)
   val05 = request.POST.get('val5')
   val5 = float(val05)
   val06 = request.POST.get('val6')
   val6 = float(val06)
   val07 = request.POST.get('val7')
   val7 = float(val07)
   val08 = request.POST.get('val8')
   val8 = float(val08)
   val09 = request.POST.get('val9')
   val9 = float(val09)
   val010 = request.POST.get('val10')
   val10 = float(val010)

   #normalize input
   val1 = (val1 - (-9.599)) / (10.666 - (-9.599))
   val2 = (val2 - (-103.384)) / (71.316 - (-103.384))
   val3 = (val3 - (-421.105)) / (394.078 - (-421.105))
   val4 = (val4 - (-301.826)) / (67.751 - (-301.826))
   val5 = (val5 - (-1.345)) / (1.197 - (-1.345))
   val6 = (val6 - (-114.686)) / (87.55 - (-114.686))
   val7 = (val7 - (-57605.281)) / (38894.4 - (-57605.281))
   val8 = (val8 - (-1127.534)) / (35.027 - (-1127.534))
   val9 = (val9 - (-355915832.7)) / (28233130.49 - (-355915832.7))
   val10 = (val10 - (-1)) / (1 - (-1))

   lis.append(val1)
   lis.append(val2)
   lis.append(val3)
   lis.append(val4)
   lis.append(val5)
   lis.append(val6)
   lis.append(val7)
   lis.append(val8)
   lis.append(val9)
   lis.append(val10)
   print(lis)

   data_array = np.asarray(lis)
   arr= data_array.reshape(1,-1)
   print(arr)

   ans = cls.predict(arr)
   print(ans)

   finalans=''
   if(ans==1):
      finalans='Your nanoparticle is NON-TOXIC'
   elif(ans==0):
      finalans = 'Your nanoparticle is TOXIC'
   print(finalans)
   return render(request, "nanotest/result.html",{'ans':finalans,"val1":val01, "val2":val02,"val3":val03, "val4":val04,
                                                  "val5":val05, "val6":val06,"val7":val07, "val8":val08,"val9":val09, "val10":val010,})

                        


