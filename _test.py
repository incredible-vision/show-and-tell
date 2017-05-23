#
# import torch
# from torch.autograd import Variable
# #
# # a=Variable(torch.FloatTensor(2,15))
# # print a
# #
# # print a.size(1)
# #
# # b=Variable(torch.zeros(2,5))
# # print b
# #
# #
# # c=Variable(torch.FloatTensor(2,3)).cuda() #Long X
# # d=Variable(torch.zeros(2,5)).cuda()
# # e = torch.cat((c,d),1)
# # print e # 2x8
#
#
# a = Variable(torch.FloatTensor([[1,2,3,4,5,6,7,8,9,10],[2,2,2,2,2,2,2,2,2,2]]))
# #print a
# H = Variable(torch.zeros(2,10,256))
#
# #print a.size()
# a= a.unsqueeze(2)
# b= a.repeat(1,1,256)
# print b.size()
# #print b
#
# context = b*H
# #print context
#
# d=torch.sum(context, dim=1)
# print d
# print d.squeeze(1)
#
#
# #c= torch.transpose(b, 0,2)
# #print c
#
# #print c*H
#
#


raise 'keywwrr'

print 1