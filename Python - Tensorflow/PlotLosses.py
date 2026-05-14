import matplotlib.pyplot as plt
from numpy import loadtxt

# Need to run "NumberDenoiseStill.py" or "NumberDenoiseInvar(0-3).py"
# first to generate results in "LossStill.txt" or "LossInvar.txt" and
# "ValLossStill.txt" or "ValLossInvar.txt".

post_fix = 'Invar'
# post_fix = 'Still'

loss_list = loadtxt('Loss' + post_fix + '.txt')
val_loss_list = loadtxt('ValLoss' + post_fix + '.txt')

plt.plot(range(1, 1+len(loss_list)), loss_list, label='loss')
plt.plot(range(1, 1+len(loss_list)), val_loss_list, label='val_loss')
plt.legend(fontsize=14)
plt.xlabel('epochs', fontsize=14)
plt.ylabel('MSE losses', fontsize=14)
plt.grid()
plt.savefig('Losses.png')
plt.show()