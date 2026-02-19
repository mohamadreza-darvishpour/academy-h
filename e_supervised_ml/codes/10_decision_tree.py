'''
Docstring for 10_decision_tree


http://scikit-learn.org/stable/modules/tree.html



روش های ارزیابی
ماتریس در هم ریختگی
confusion matris
tp,tn, fp,fn    حالات متفاوت پیشبینی

score.


دقت در راستای پیشبینی های مثبت
precision = tp /(tp+fp)


در راستای برچسب های مثبت
recall = tp/(tp+fn)

F1 = 2*precision*recall / (precision + recall)

در راستای این که چند تاش درست پیشبینی شده.
accuracy = (tp+tn)/(tp+fn + tn + fp)

specificity = tn / (tn + fp)

roc نمودار
true pos on false pos chart


این معیارها برای مجموعه داده ی آزمووون هستن نه تمرین.
'''

'''
defining model evaluation rules 
http://scikit-learn.org/stable/modules/model_evaluation.html

'''

