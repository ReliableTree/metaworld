from __future__ import absolute_import, division, print_function, unicode_literals
from functools import total_ordering
from os import name, path, makedirs
from random import seed
from xml.etree.ElementTree import QName
from cv2 import add
from imitation import data
import tensorflow as tf
import sys
import numpy as np
from LanguagePolicies.utils.graphsTorch import TBoardGraphsTorch
from LanguagePolicies.model_src.modelTorch import PolicyTranslationModelTorch
from MetaWorld.utilsMW.metaOptimizer import SignalModule, TaylorSignalModule, meta_optimizer, tailor_optimizer, MetaModule
from MetaWorld.utilsMW.dataLoaderMW import TorchDatasetMW, TorchDatasetTailor
from imitation.data import rollout
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data.wrappers import RolloutInfoWrapper
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from MetaWorld.searchTest.utils import sample_expert_transitions, VecExtractor, parse_sampled_transitions, HER_Transitions, new_epoch_np, make_rollouts_vec
from MetaWorld.utilsMW.activate_critic_policy import ActiveCriticPolicy
import copy
import os
import time
from gym.envs.mujoco import MujocoEnv
from MetaWorld.utilsMW.model_setup_obj import ActiveCriticArgs
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class NetworkTrainer(nn.Module):
    def __init__(self, model:PolicyTranslationModelTorch, 
    tailor_models, 
    env_tag, 
    env:MujocoEnv, 
    data_path, 
    logname, 
    lr, 
    mlr, 
    mo_lr, 
    gamma_sl = 0.995, 
    device = 'cuda', 
    tboard=True,
    network_args_obj:ActiveCriticArgs=None
    ):
        super().__init__()
        self.network_args = network_args_obj
        self.optimizer         = None
        self.model             = model
        self.tailor_models      = torch.nn.ModuleList(tailor_models)
        self.total_steps       = 0
        self.logname           = logname
        self.lr = lr
        self.mlr = mlr
        self.mo_lr = mo_lr
        self.device = device
        self.data_path = data_path
        self.use_tboard = tboard
        self.embedding_memory = {}
        self.env_tag = env_tag
        self.init_train = True
        self.max_success_rate = 0
        self.max_step_disc = 1200 * 16
        
        self.extractor = network_args_obj.extractor

        if self.logname.startswith("Intel$"):
            self.instance_name = self.logname.split("$")[1]
            self.logname       = self.logname.split("$")[0]
        else:
            self.instance_name = None

        if tboard:
            self.tboard            = TBoardGraphsTorch(self.logname, data_path=data_path)
        self.loss              = nn.CrossEntropyLoss()
        self.global_best_loss  = float('inf')
        self.global_best_loss_val = float('inf')
        self.last_written_step = -1

        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.gamma_sl = gamma_sl

        self.last_mean_success = 0

        self.best_improved_success = 0
        self.global_step = 0
        self.add_data = not network_args_obj.imitation_phase

        self.env = env
        self.num_vals = 0


    def setup_model(self, device = 'cuda'):
        def lfp(result, label):
            return ((result.reshape(-1)-label.reshape(-1))**2).mean()
        self.tailor_modules = []

        for i in range(len(self.tailor_models)):
            self.tailor_modules.append(TaylorSignalModule(model=self.tailor_models[i], loss_fct=lfp, lr = self.mlr, mlr = self.mlr))

        self.model_state_dict = self.model.state_dict()
        self.signal_main = SignalModule(model=self.model, loss_fct=self.calculateLoss)

        self.policy = ActiveCriticPolicy(
            observation_space=self.env.observation_space, 
            action_space=self.env.action_space, 
            main_signal=self.signal_main,
            tailor_signals=self.tailor_modules,
            lr = self.mo_lr,
            writer=self.write_tboard_scalar,
            args_obj=self.network_args)

        self.policy.return_mode = 1

        env = self.env()
        obsv = env.reset().reshape([1,1,-1])
        obsv = obsv.repeat(100, axis=1)
        _ = self.policy.predict(observation=obsv)
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=self.network_args.weight_decay) 
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 40, self.gamma_sl, verbose=False)
        self.trajectories=None

    def loadTailorDataset(self, path):
        obs_path = path + 'obs'
        trj_path = path + 'trj'
        s_path = path + 'success'
        if os.path.exists(s_path):
            self.inpt_obs = torch.load(obs_path)
            self.trajectories = torch.load(trj_path)
            self.success = torch.load(s_path)

    def setDatasets(self, train_loader:DataLoader, val_loader:DataLoader, batch_size = 32):
        self.train_loader = train_loader
        train_ds = train_loader.dataset
        obsvs = train_ds.data
        actions = train_ds.label
        reward = torch.ones(size=[len(obsvs)], dtype=torch.bool, device=actions.device)
        tailor_data = TorchDatasetTailor(trajectories=actions, obsv=obsvs, success=reward, ftrj=actions)
        self.tailor_loader = DataLoader(dataset=tailor_data, batch_size=batch_size, shuffle=True)
        self.val_loader   = val_loader
        

    def train_tailor(self, ):
        trajectories, inpt_obs, success, ftrjs = self.successSimulation(policy = self.model, env_tag = self.env_tag, n = 10)
        inpt = torch.concat((trajectories, inpt_obs), dim = 0)
        debug_dict = tailor_optimizer(tailor_modules=self.tailor_modules, inpt=inpt, label=success)
        self.write_tboard_train_scalar(debug_dict=debug_dict)


    def train(self, epochs):
        self.expert_examples_len = len(self.train_loader.dataset)
        print(f'inital num examples: {self.expert_examples_len}')
        disc_epoch = 0
        disc_step = 0
        model_step = 0
        for epoch in range(epochs):
            if self.add_data:
                policy = self.policy
                policy.return_mode = 1
                seed = np.random.randint(0, 1e10, 1)
                self.sample_new_episode(policy=policy, episodes=1, seeds=seed)
            while model_step < self.network_args.n_steps:
                #print("Epoch: {:3d}/{:3d}".format(epoch+1, epochs)) 
                validation_loss = 0.0
                train_loss = []
                if not self.init_train and False:
                    self.policy.train()
                    #self.model.eval()
                    loss_module = 1
                    disc_epoch += 1
                    lmp = None
                    lmn = None
                    disc_step += len(self.tailor_loader.dataset)
                    for data in self.tailor_loader:

                        self.global_step += 1
                        debug_dict = self.tailor_step(data)
                        if lmp is None:
                            lmp = debug_dict['tailor loss positive'].reshape(1)
                            lmn = debug_dict['tailor loss positive'].reshape(1)
                        else:
                            lmp = torch.cat((lmp, debug_dict['tailor loss positive'].reshape(1)), 0)
                            lmn = torch.cat((lmn, debug_dict['tailor loss negative'].reshape(1)), 0)

                        loss_module = torch.maximum(lmp, lmn).mean()
                    
                    debug_dict['tailor module loss'] = loss_module
                    self.write_tboard_scalar(debug_dict=debug_dict, train=True)

                
                self.model.train()
                self.policy.train()

                model_step += len(self.train_loader.dataset)
                for step, (d_in, d_out) in enumerate(self.train_loader):
                    train_loss.append(self.step(d_in, d_out, train=True))
                    if epoch == 0:
                        self.total_steps += 1
                    self.global_step += 1

                    #self.loadingBar(self.total_steps, self.total_steps, 25, addition="Loss: {:.6f}".format(np.mean(train_loss)), end=True)
            disc_step = 0
            model_step = 0
            self.num_vals += 1
            complete = (self.num_vals%self.network_args.complete_modulo == 0)
            print(f'logname: {self.logname}')
            if complete:
                self.runValidation(quick=False, epoch=epoch, save=True, complete=False)            


            self.scheduler.step()


    def sample_new_episode(self, policy:ActiveCriticPolicy, episodes:int, seeds:list[int], add_data = True, her = True):
        num_cpu = self.network_args.num_cpu
        if num_cpu > episodes:
            num_cpu = episodes

        #num_cpu, env_constr, epoch_len, seed        
        transitions = make_rollouts_vec(seeds=seeds, expert=policy, env_constr=self.env, num_cpu=num_cpu, epoch_len=self.network_args.epoch_len)

        transitions = sample_expert_transitions(policy, self.env(), episodes)
        if her:
            transitions = HER_Transitions(transitions=transitions, new_epoch=new_epoch_np)
        datas  = parse_sampled_transitions(transitions=transitions, new_epoch=self.network_args.new_epoch, extractor=self.extractor)
        device_data = []
        for data in datas:
            device_data.append(data.to(self.network_args.device))
        actions, observations, rewards = device_data
        actions = actions[:episodes]
        observations = observations[:episodes]
        rewards = rewards[:episodes]
        if add_data:
            self.add_data_to_loader(inpt_obs_opt=observations, trajectories_opt=actions, success_opt=rewards, ftrjs_opt=actions, episodes=episodes)
        return actions, observations, rewards
    
    def tailor_step(self, data):
        debug_dict = tailor_optimizer(tailor_modules = self.tailor_modules, data=data)
        self.write_tboard_scalar(debug_dict=debug_dict, train=True)
        return debug_dict
    
    def torch2tf(self, inpt):
        if inpt is not None:
            return tf.convert_to_tensor(inpt.detach().cpu().numpy())
        else:
            return None
    def tf2torch(self, inpt):
        if inpt is not None:
            return torch.tensor(inpt.numpy(), device= self.device)
        else:
            return None

    def runvalidationTaylor(self, data, debug_dict={}, return_mode = 0):
        actions, observations, success = data
        fail = ~success
        taylor_inpt = {'result':actions, 'inpt':observations, 'original':actions}

        for i, ts in enumerate(self.policy.tailor_signals):
            expected_success = ts.forward(taylor_inpt)
            #expected_success = self.tailor_modules[0].forward(taylor_inpt)

            expected_success = expected_success.reshape(-1).type(torch.bool)
            expected_fail = ~ expected_success
            expected_success = expected_success.type(torch.float)
            expected_fail = expected_fail.type(torch.float)

            fail = fail.type(torch.float).reshape(-1)
            success = success.type(torch.float).reshape(-1)
            tp = (expected_success * success)[success==1].mean()
            if success.sum() == 0:
                tp = torch.tensor(0)
            fp = (expected_success * fail)[fail==1].mean()
            tn = (expected_fail * fail)[fail==1].mean()
            fn = (expected_fail * success)[success==1].mean()

            if return_mode == 0:
                add = ' '+str(i)
            elif return_mode == 1:
                add = ' optimized '+str(i)
            elif return_mode == 2:
                add = ' label '+str(i)

            debug_dict['true positive' + add] = tp
            debug_dict['false positive' + add] = fp
            debug_dict['true negative' + add] = tn
            debug_dict['false negative' + add] = fn
            debug_dict['tailor success' + add] = (expected_success==success).type(torch.float).mean()
            debug_dict['tailor expected success' + add] = (expected_success).type(torch.float).mean()

        return debug_dict

    def runValidation(self, quick=False, pnt=True, epoch = 0, save = False, complete = False): 
        self.policy.eval()
        self.model.eval()
        num_examples = self.tailor_loader.dataset._num_elements()
        #with torch.no_grad():
        if (not quick):
            if complete:
                num_envs = self.network_args.eval_epochs
            else:
                num_envs = self.network_args.eval_epochs
            
            #torch.manual_seed(1)
            print("Running full validation...")

            policy = self.policy
            policy.main_signal.model.eval()
            policy.return_mode = 0
            num_seeds = 1+num_envs//self.network_args.num_cpu
            seeds = np.random.randint(0, 1e10, num_seeds)
            actions, observations, success = self.sample_new_episode(policy=policy, env=self.env, episodes=num_envs, seeds=seeds, add_data=False, her=False)
            data_gen = (actions, observations, success.type(torch.bool))
            print(f'num envs: {len(actions)}')
            mean_success = success.mean()
            fail = ~success.type(torch.bool)
            print(f'mean success before: {mean_success}')
            debug_dict = {'success rate generated' : mean_success}
            self.write_tboard_scalar(debug_dict=debug_dict, train=False)
            #TODO:
            #Zurück ändern!
            policy.return_mode = 0
            actions_opt, observations_opt, success_opt = self.sample_new_episode(policy=policy, env=self.env, episodes=num_envs, seeds=seeds, add_data=False, her=False)
            data_opt = (actions_opt, observations_opt, success_opt.type(torch.bool))
            assert torch.equal(observations[0,0], observations_opt[0,0]), "environments changed"

            if complete:
                print('complete:')
                self.policy.optim_run += 1
                debug_dict = self.runvalidationTaylor(data=data_gen, return_mode=0)
                self.write_tboard_scalar(debug_dict=debug_dict, train = False)
                debug_dict = self.runvalidationTaylor(data=data_opt, return_mode=1)
                self.write_tboard_scalar(debug_dict=debug_dict, train = False)

            
            
            if len(success_opt)>0:
                mean_success_opt = success_opt.mean()
            else:
                mean_success_opt = 0
            if mean_success_opt > self.last_mean_success:
                policy.max_steps = policy.max_steps * 1.1
                self.last_mean_success = mean_success_opt

            fail_opt = ~success_opt.type(torch.bool)

            
            '''if tailor_success_optimized * 0.8 > mean_success_opt:
                policy.max_steps = policy.max_steps * 1.1
            elif tailor_success_optimized < mean_success_opt:
                policy.max_steps = max(policy.max_steps * 0.9, 5)'''

            self.write_tboard_scalar({'num optimisation steps':torch.tensor(policy.max_steps)}, train= False)

            print(f'mean success after: {mean_success_opt}')
            debug_dict = {}
            debug_dict['success rate optimized'] = mean_success_opt
            debug_dict['improved success rate'] = mean_success_opt - mean_success
            if mean_success_opt - mean_success > self.best_improved_success:
                self.best_improved_success = mean_success_opt - mean_success
                self.saveNetworkToFile(add=self.logname + "/best_improved/", data_path= self.data_path)

            num_improved = (success_opt * fail).type(torch.float).mean()
            num_deproved = (success * fail_opt).type(torch.float).mean()
            debug_dict['rel number improved'] = num_improved
            debug_dict['rel number failed'] = num_deproved

            self.write_tboard_scalar(debug_dict=debug_dict, train=not complete, step = self.global_step)
        
            if mean_success > self.max_success_rate:
                self.max_success_rate = mean_success
            else:
                pass
                #self.model.load_state_dict(self.model_state_dict)            

        val_loss = []
        for step, (d_in, d_out) in enumerate(self.val_loader):
            loss = self.step(d_in, d_out, train=False)
            val_loss.append(loss)
            if quick:
                break

        if self.use_tboard:
            do_dim = d_in[0].size(0)
            #print(d_in)
            #self.model.eval()
            
            if not quick:
                success = success.type(torch.bool)
                success_opt = success_opt.type(torch.bool)
                self.plot_with_mask(label=actions, trj=actions, inpt=observations, mask=success, name = 'success')

                fail = ~success
                fail_opt = ~success_opt
                self.plot_with_mask(label=actions, trj=actions, inpt=observations, mask=fail, name = 'fail')

                fail_to_success = success_opt & fail
                self.plot_with_mask(label=actions, trj=actions, inpt=observations, mask=fail_to_success, name = 'fail to success', opt_trj=actions_opt)

                success_to_fail = success & fail_opt
                self.plot_with_mask(label=actions, trj=actions, inpt=observations, mask=success_to_fail, name = 'success to fail', opt_trj=actions_opt)
                
                fail_to_fail = fail & fail_opt
                self.plot_with_mask(label=actions, trj=actions, inpt=observations, mask=fail_to_fail, name = 'fail to fail', opt_trj=actions_opt)

                
                #print(f'label-check: {label[success_to_fail][0]}')
                #print(f'label-opt-check: {label_opt[success_to_fail][0]}')



        loss = np.mean(val_loss)
        if pnt:
            print("  Validation Loss: {:.6f}".format(loss))
        if not quick:
            if loss < self.global_best_loss_val:
                self.global_best_loss_val = loss
            if self.use_tboard:
                self.saveNetworkToFile(add=self.logname + "/last/", data_path= self.data_path)
        return np.mean(val_loss)


    def add_data_to_loader(self, inpt_obs_opt, trajectories_opt, success_opt, ftrjs_opt, episodes):
        if self.add_data:

            print(f'inpt_obs_opt: {inpt_obs_opt.shape}')
            print(f'trajectories_opt: {trajectories_opt.shape}')
            print(f'success_opt: {success_opt}')
            print(f'ftrjs_opt: {ftrjs_opt.shape}')
            print(f'episodes: {episodes}')

            inpt_obs_opt = inpt_obs_opt.to(self.network_args.device)
            trajectories_opt = trajectories_opt.to(self.network_args.device)
            success_opt = success_opt.to(self.network_args.device)
            ftrjs_opt = ftrjs_opt.to(self.network_args.device)
                
            train_data = self.train_loader.dataset
            if success_opt[:episodes].sum() > 0:
                success_opt = success_opt.type(torch.bool)
                train_data.add_data(data=inpt_obs_opt[:episodes][success_opt[:episodes]], label=trajectories_opt[:episodes][success_opt[:episodes]])
                #self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
            
            tailor_data = self.tailor_loader.dataset
            tailor_data.add_data(trajectories=trajectories_opt, obsv=inpt_obs_opt, success=success_opt, ftrj=ftrjs_opt)
            self.write_tboard_scalar({'num examples':torch.tensor(len(tailor_data))}, train=False)
            self.init_train = ~tailor_data.success.sum() == 0
        print(f'num examples: {self.tailor_loader.dataset._num_elements()}')
        print(f'num demonstrations: {len(self.train_loader.dataset)}')

    def write_tboard_scalar(self, debug_dict, train, step = None):
        if step is None:
            step = self.global_step
        if self.use_tboard:
                for para, value in debug_dict.items():
                    value = value.to('cpu')
                    if train:
                        self.tboard.addTrainScalar(para, value, step)
                    else:
                        self.tboard.addValidationScalar(para, value, step)

    def plot_with_mask(self, label, trj, inpt, mask, name, opt_trj=None):
        if mask.sum()>0:
            label = label[mask][0]
            trj = trj[mask][0]
            inpt = inpt[mask][0,0]
            if opt_trj is not None:
                opt_trj = opt_trj[mask][0]
            self.createGraphsMW(d_in=1, d_out=label, result=trj, toy=False, inpt=inpt, name=name, opt_trj=opt_trj, window=0)

    def step(self, d_in, d_out, train):

        if train:
            result = self.model(d_in)
            loss, debug_dict = self.calculateLoss(d_out, result)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.write_tboard_scalar(debug_dict=debug_dict, train=True)
        else:
            with torch.no_grad():
                result = self.model(d_in)
                loss, debug_dict = self.calculateLoss(d_out=d_out, result=result)
            

            if self.last_written_step != self.global_step:
                if self.use_tboard:
                    self.last_written_step = self.global_step
                    self.write_tboard_scalar(debug_dict=debug_dict, train=False)

                loss = loss.detach().cpu()
                if loss < self.global_best_loss:
                    self.global_best_loss = loss
                    #print(f'model saved with loss: {loss}')

        return loss.detach().cpu().numpy()
    
    def interpolateTrajectory(self, trj, target):
        batch_size     = trj.shape[0]
        current_length = trj.shape[1]
        dimensions     = trj.shape[2]
        result         = np.zeros((batch_size, target, dimensions), dtype=np.float32)
    
        for b in range(batch_size):
            for i in range(dimensions):
                result[b,:,i] = np.interp(np.linspace(0.0, 1.0, num=target), np.linspace(0.0, 1.0, num=current_length), trj[b,:,i])
        
        return result

    def calculateMSEWithPaddingMask(self, y_true, y_pred, mask):
        mse = (y_true - y_pred)**2
        mse = mse * mask
        return mse.mean(-1)

    def catCrossEntrLoss(self, y_labels, y_pred):
        y_labels_args = torch.argmax(y_labels, dim = -1)
        return nn.NLLLoss()(torch.log(y_pred), y_labels_args)

    def calcMSE(self, a, b):
        return ((a.squeeze() - b.squeeze())**2).mean()

    def calculateLoss(self, d_out, result, prefix = ''):
        generated = d_out

        gen_trj = result['gen_trj']
        #phs = result['phs']

        #phs_loss = self.calcMSE(phase, phs)
        trj_loss = self.calcMSE(generated, gen_trj)

        debug_dict = {
            prefix + 'trj_loss':trj_loss,
            #prefix + 'phs_loss':phs_loss,
            }
        loss = trj_loss * self.lr
        debug_dict[prefix + 'main_loss'] = loss
        
        return (loss, debug_dict)
    
    def loadingBar(self, count, total, size, addition="", end=False):
        if total == 0:
            percent = 0
        else:
            percent = float(count) / float(total)
        full = int(percent * size)
        fill = size - full
        print("\r  {:5d}/{:5d} [".format(count, total) + "#" * full + " " * fill + "] " + addition, end="")
        if end:
            print("")
        sys.stdout.flush()

    def createGraphs(self, d_in, d_out, result, save = False, name_plot = '', epoch = 0):
        language, image, robot_states            = d_in
        target_trj, attention, delta_t, weights  = d_out
        gen_trj = result['gen_trj']
        atn     = result['atn']
        phase   = result['phs']


        self.tboard.plotClassAccuracy(attention, self.tf2torch(tf.math.reduce_mean(self.torch2tf(atn), axis=0)), self.tf2torch(tf.math.reduce_std(self.torch2tf(atn), axis=0)), self.tf2torch(self.torch2tf(language)), stepid=self.global_step)
        path_to_plots = self.data_path + "/plots/"+ str(self.logname) + '/' + str(epoch) + '/'
        gen_tr_trj= self.tf2torch(tf.math.reduce_mean(self.torch2tf(gen_trj), axis=0))
        gen_tr_phase = self.tf2torch(tf.math.reduce_mean(self.torch2tf(phase), axis=0))
        self.tboard.plotDMPTrajectory(target_trj, gen_tr_trj, torch.zeros_like(gen_tr_trj),
                                    gen_tr_phase, delta_t, None, stepid=self.global_step, save=save, name_plot=name_plot, path=path_to_plots)
        
    def createGraphsMW(self, d_in, d_out, result, save = False, name_plot = '', epoch = 0, toy=True, inpt=None, name='Trajectory', opt_trj = None, window = 0):
        target_trj  = d_out
        gen_trj = result
        
        path_to_plots = self.data_path + "/plots/"+ str(self.logname) + '/' + str(epoch) + '/'

        tol_neg = None
        tol_pos = None
        #gen_tr_trj= self.tf2torch(tf.math.reduce_mean(self.torch2tf(gen_trj), axis=0))
        #gen_tr_phase = self.tf2torch(tf.math.reduce_mean(self.torch2tf(phase), axis=0))
        self.tboard.plotDMPTrajectory(target_trj, gen_trj, torch.zeros_like(gen_trj),
                                    None, None, None, stepid=self.global_step, save=save, name_plot=name_plot, path=path_to_plots,\
                                        tol_neg=tol_neg, tol_pos=tol_pos, inpt = inpt, name=name, opt_gen_trj = opt_trj, window=window)
        

    def saveNetworkToFile(self, add, data_path):
        import pickle
        import os
        #dir_path = path.dirname(path.realpath(__file__))
        path_to_file = os.path.join(data_path, "Data/Model/", add)
        if not path.exists(path_to_file):
            makedirs(path_to_file)

        torch.save(self.state_dict(), path_to_file + "policy_network")
        torch.save(self.tailor_modules[0].model.state_dict(), path_to_file + "tailor_network")
        torch.save(self.optimizer.state_dict(), path_to_file + "optimizer")
        torch.save(self.tailor_modules[0].meta_optimizer.state_dict(), path_to_file + "tailor_optimizer")
        torch.save(torch.tensor(self.global_step), path_to_file + "global_step")


        with open(path_to_file + 'model_setup.pkl', 'wb') as f:
            pickle.dump(self.network_args, f)  

        torch.save(self.tailor_loader, path_to_file+'tailor')
        torch.save(self.train_loader, path_to_file+'train')
        
        '''tailor_obs = ['obs', 'trj', 'success', 'ftrj']
        tailor_data = [self.inpt_obs, self.trajectories, self.success, self.ftrj]
        for i, name in enumerate(tailor_obs):
            torch.save(tailor_data[i], path_to_file+name)'''

    def loadNetworkFromFile(self, path, device = 'cuda'):
        self.load_state_dict(torch.load(path + "policy_network", map_location=device))
        self.tailor_modules[0].model.load_state_dict(torch.load(path + "tailor_network", map_location=device))
        self.optimizer.load_state_dict(torch.load(path + "optimizer"))
        self.tailor_modules[0].meta_optimizer.load_state_dict(torch.load(path + "tailor_optimizer", map_location=device))
        
        '''self.inpt_obs = torch.load(path+'obs', map_location=device)
        self.trajectories = torch.load(path+'trj', map_location=device)
        self.success = torch.load(path+'success', map_location=device).type(torch.bool)
        self.ftrj = torch.load(path+'ftrj', map_location=device)'''

        self.global_step = int(torch.load(path+'global_step'))

        self.tailor_loader = torch.load(path+'tailor')
        self.train_loader = torch.load(path+'train')

        '''tailor_data = TorchDatasetTailor(trajectories= self.trajectories, obsv=self.inpt_obs, success=self.success, ftrj = self.ftrj)
        self.tailor_loader = DataLoader(tailor_data, batch_size=32, shuffle=True)

        train_data = self.train_loader.dataset
        train_data.set_data(data=self.inpt_obs[self.success], label=self.trajectories[self.success])
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)'''

