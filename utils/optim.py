import torch
import logging.config
from copy import deepcopy
from six import string_types


def eval_func(f, x):
    if isinstance(f, string_types):
        f = eval(f)
    return f(x)


class OptimRegime(object):
    """
    Reconfigures the optimizer according to setting list.
    Exposes optimizer methods - state, step, zero_grad, add_param_group

    Examples for regime:

    1)  "[{'epoch': 0, 'optimizer': 'Adam', 'lr': 1e-3},
          {'epoch': 2, 'optimizer': 'Adam', 'lr': 5e-4},
          {'epoch': 4, 'optimizer': 'Adam', 'lr': 1e-4},
          {'epoch': 8, 'optimizer': 'Adam', 'lr': 5e-5}
         ]"
    2)
        "[{'step_lambda':
            "lambda t: {
            'optimizer': 'Adam',
            'lr': 0.1 * min(t ** -0.5, t * 4000 ** -1.5),
            'betas': (0.9, 0.98), 'eps':1e-9}
         }]"
    """

    def __init__(self, params, regime):
        self.optimizer = torch.optim.SGD(params, lr=0)
        self.regime = regime
        self.current_regime_phase = None
        self.setting = {}

    def update(self, epoch, train_steps):
        """adjusts optimizer according to current epoch or steps and training regime.
        """
        if self.regime is None:
            return
        update_optimizer = False
        if self.current_regime_phase is None:
            update_optimizer = True
            setting = {}
            # Find the first entry where the epoch is smallest than current
            for regime_phase, regime_setting in enumerate(self.regime):
                start_epoch = regime_setting.get('epoch', 0)
                start_step = regime_setting.get('step', 0)
                if epoch >= start_epoch or train_steps >= start_step:
                    self.current_regime_phase = regime_phase
                    break
        if len(self.regime) > self.current_regime_phase + 1:
            next_phase = self.current_regime_phase + 1
            # Any more regime steps?
            start_epoch = self.regime[next_phase].get('epoch', float('inf'))
            start_step = self.regime[next_phase].get('step', float('inf'))
            if epoch >= start_epoch or train_steps >= start_step:
                self.current_regime_phase = next_phase
                update_optimizer = True

        setting = deepcopy(self.regime[self.current_regime_phase])

        if 'lr_decay_rate' in setting and 'lr' in setting:
            decay_steps = setting.get('lr_decay_steps', 100)
            if train_steps % decay_steps == 0:
                decay_rate = setting['lr_decay_rate']
                setting['lr'] *= decay_rate ** (train_steps / decay_steps)
                update_optimizer = True
        elif 'step_lambda' in setting:
            setting.update(eval_func(setting['step_lambda'], train_steps))
            update_optimizer = True
        elif 'epoch_lambda' in setting:
            setting.update(eval_func(setting['epoch_lambda'], epoch))
            update_optimizer = True

        if update_optimizer:
            self.adjust(setting)

    def adjust(self, setting):
        """adjusts optimizer according to a setting dict.
        e.g: setting={optimizer': 'Adam', 'lr': 5e-4}
        """
        if 'optimizer' in setting:
            optim_method = torch.optim.__dict__[setting['optimizer']]
            if not isinstance(self.optimizer, optim_method):
                self.optimizer = optim_method(self.optimizer.param_groups)
                logging.debug('OPTIMIZER - setting method = %s' %
                              setting['optimizer'])
        for param_group in self.optimizer.param_groups:
            for key in param_group.keys():
                if key in setting:
                    new_val = setting[key]
                    if new_val != param_group[key]:
                        logging.debug('OPTIMIZER - setting %s = %s' %
                                      (key, setting[key]))
                        param_group[key] = setting[key]
        self.setting = deepcopy(setting)

    def __getstate__(self):
        return {
            'optimizer_state': self.optimizer.__getstate__(),
            'regime': self.regime,
        }

    def __setstate__(self, state):
        self.regime = state.get('regime')
        self.optimizer.__setstate__(state.get('optimizer_state'))

    def state_dict(self):
        """Returns the state of the optimizer as a :class:`dict`.
        """
        return {
            'optimizer_state': self.optimizer.state_dict(),
            'regime': self.regime,
        }

    def load_state_dict(self, state_dict):
        """Loads the optimizer state.

        Arguments:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        optimizer_state_dict = state_dict['optimizer_state']

        self.__setstate__({'optimizer_state': optimizer_state_dict,
                           'regime': state_dict['regime']})

    def zero_grad(self):
        """Clears the gradients of all optimized :class:`Variable` s."""
        self.optimizer.zero_grad()

    def step(self, closure=None):
        """Performs a single optimization step (parameter update).

        Arguments:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        self.optimizer.step(closure)

    def add_param_group(self, param_group):
        """Add a param group to the :class:`Optimizer` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.

        Arguments:
            param_group (dict): Specifies what Variables should be optimized along with group
            specific optimization options.
        """
        self.optimizer.add_param_group(param_group)
