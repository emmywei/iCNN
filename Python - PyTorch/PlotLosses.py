import matplotlib.pyplot as plt
import torch

# Need to run "NumberDenoiseStill.py" or "NumberDenoiseInvar(0-3).py"
# first to generate results in "LossStill.txt" or "LossInvar.txt" and
# "ValLossStill.txt" or "ValLossInvar.txt".

post_fix = 'Invar'
# post_fix = 'Still'

loss_list = torch.tensor(
    [float(x.strip()) for x in open('Loss' + post_fix + '.txt')]
)

val_loss_list = torch.tensor(
    [float(x.strip()) for x in open('ValLoss' + post_fix + '.txt')]
)

plt.plot(range(1, 1 + len(loss_list)), loss_list.numpy(), label='loss')
plt.plot(range(1, 1 + len(val_loss_list)), val_loss_list.numpy(), label='val_loss')
plt.legend(fontsize=14)
plt.xlabel('epochs', fontsize=14)
plt.ylabel('MSE losses', fontsize=14)
plt.grid()
plt.savefig('Losses.png')
plt.show()