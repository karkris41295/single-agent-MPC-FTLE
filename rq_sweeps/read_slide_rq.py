import numpy as np
import matplotlib.pyplot as plt

recall = np.load('t4up1delp1.npz')
data = [recall[key] for key in recall]
A, eps, omega, Delta, Nsim, Nt, Nu, Nx, es, es2, goal, preds, qr, tc, ubot, uru = data
#A, Delta, Nsim, Nt, Nu, Nx, es, es2, goal, preds, qr, tc, ubot, uru = data

# %% Moving bundle plot

flag = 1

from matplotlib import animation

if flag == 1: #if points were spaced equally
    fig = plt.figure()
    ax = plt.subplot(211, aspect = 'equal', xlim=(0, 2), ylim=(0, 1), title = 'MPC horizon bundle, $T_{horizon} = $' + str(Nt*Delta) + ', $u_{botmax} = $' + str(ubot))
    ax = plt.subplot(211, aspect = 'equal', xlim=(0, 2), ylim=(0, 1))
    ax2 = plt.subplot(212, title = 'costs', xlabel = 'R/Q', xlim=(0, max(qr)))
    plt.tight_layout()
    #qr_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    cs2 = tc-uru
    ax2.plot(qr,uru*Delta, 'grey', alpha = .3)
    ax2.plot(qr,cs2*Delta, 'grey', alpha = .3)
    ax2.plot(qr,tc*Delta, 'grey', alpha = .3)
    particle1, = ax2.plot([], [], marker='o', linewidth = 0, color='black')
    particle2, = ax2.plot([], [], marker='o', linewidth = 0, color='black')
    particle3, = ax2.plot([], [], marker='o', linewidth = 0, color='black')
    traj1, = ax2.plot([],[], color='blue', alpha=0.6, label = '$\sum (u^TRu) \Delta t$')
    traj2, = ax2.plot([],[], color='red', alpha=0.6, label = '$\sum (x^TQx) \Delta t$')
    traj3, = ax2.plot([],[], color='purple', alpha=0.6, label = '$\sum (x^TQx + u^TRu) \Delta t$')
    ax2.legend()
    
    from matplotlib import cm
    
    def update(num):
        '''
        updates animation
        '''
        ax.cla()
        ax.set_xlim(0,2)
        ax.set_ylim(0,1)
        #ax.set_title('MPC horizon bundle, $T_{horizon} = $' + str(Nt*Delta) + ', $u_{botmax} = $' + str(ubot))
        qr_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
        ax.scatter(goal[0], goal[1], marker = 'X', color='green', s=90)
        e = es[num]
        for i in range(Nsim):
            ax.plot(preds[num][i,:,0], preds[num][i,:,1], alpha=0.02, c = cm.plasma(e[i]/max(1.00001*e)))
            
        qr_text.set_text('$R/Q = $' + str(qr[num]))
        
        particle1.set_data(qr[num], uru[num]*Delta)
        traj1.set_data(qr[:num+1],uru[:num+1]*Delta) 
        particle2.set_data(qr[num], cs2[num]*Delta)
        traj2.set_data(qr[:num+1],cs2[:num+1]*Delta) 
        particle3.set_data(qr[num], tc[num]*Delta)
        traj3.set_data(qr[:num+1],tc[:num+1]*Delta) 
        
    anim = animation.FuncAnimation(fig, update, blit=False, interval=100)

elif flag == 2: # if points were spaced logarithmically
    from matplotlib import animation
    
    qr = 10**(np.arange(0,100,1)*(4./100) - 2)
    fig = plt.figure()
    #ax = plt.subplot(211, aspect = 'equal', xlim=(0, 2), ylim=(0, 1), title = 'MPC horizon bundle, $T_{horizon} = $' + str(Nt*Delta) + ', $u_{botmax} = $' + str(ubot))
    ax = plt.subplot(211, aspect = 'equal', xlim=(0, 2), ylim=(0, 1))
    ax2 = plt.subplot(212, title = 'costs', xlabel = '$R/Q_2$', xlim=(0, max(qr)))
    plt.tight_layout()
    #qr_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    uru = np.array(es2)*qr
    ax2.plot(qr,uru, 'grey', alpha = .3)
    ax2.plot(qr,cs2, 'grey', alpha = .3)
    ax2.plot(qr,tc, 'grey', alpha = .3)
    particle1, = ax2.plot([], [], marker='o', linewidth = 0, color='black')
    particle2, = ax2.plot([], [], marker='o', linewidth = 0, color='black')
    particle3, = ax2.plot([], [], marker='o', linewidth = 0, color='black')
    traj1, = ax2.plot([],[], color='blue', alpha=0.6, label = '$\sum (u^TRu)$')
    traj2, = ax2.plot([],[], color='red', alpha=0.6, label = '$\sum (x^TQx)$')
    traj3, = ax2.plot([],[], color='purple', alpha=0.6, label = '$\sum (x^TQx + u^TRu)$')
    ax2.legend()
    ax2.set_xscale('symlog')
    
    from matplotlib import cm
    
    def update(num):
        '''
        updates animation
        '''
        ax.cla()
        ax.set_xlim(0,2)
        ax.set_ylim(0,1)
        ax.set_title('MPC horizon bundle, $T_{horizon} = $' + str(Nt*Delta) + ', $u_{botmax} = $' + str(ubot))
        qr_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
        ax.scatter(goal[0], goal[1], color = 'purple')
        e = es[num]
        for i in range(Nsim):
            ax.plot(preds[num][i,:,0], preds[num][i,:,1], alpha=0.2, c = cm.plasma(e[i]/max(1.00001*e)))
            
        qr_text.set_text('$R/Q2 = $' + str(qr[num]))
        
        particle1.set_data(qr[num], uru[num])
        traj1.set_data(qr[:num+1],uru[:num+1]) 
        particle2.set_data(qr[num], cs2[num])
        traj2.set_data(qr[:num+1],cs2[:num+1]) 
        particle3.set_data(qr[num], tc[num])
        traj3.set_data(qr[:num+1],tc[:num+1]) 
        ax2.set_xscale('symlog')
        
    anim = animation.FuncAnimation(fig, update, blit=False, interval=100)
        
