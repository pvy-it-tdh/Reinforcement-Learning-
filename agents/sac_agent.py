# agents/sac_agent.py
import torch
import torch.nn.functional as F
import torch.optim as optim
from agents.networks import Actor, Critic


class SACAgent:
    def __init__(self, obs_dim, act_dim, device="cpu"):
        self.device = device
        self.act_dim = act_dim

        # ========== Networks ==========
        self.actor = Actor(obs_dim, act_dim).to(device)

        self.q1 = Critic(obs_dim, act_dim).to(device)
        self.q2 = Critic(obs_dim, act_dim).to(device)

        self.q1_target = Critic(obs_dim, act_dim).to(device)
        self.q2_target = Critic(obs_dim, act_dim).to(device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # ========== Optimizers ==========
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=3e-4)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=3e-4)

        # ========== Entropy (alpha) ==========
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=3e-4)

        self.target_entropy = -act_dim

        # ========== Hyperparameters ==========
        self.gamma = 0.99
        self.tau = 0.005

    # ------------------------------------------------
    @property
    def alpha(self):
        return self.log_alpha.exp()

    # ------------------------------------------------
    @torch.no_grad()
    def act(self, obs, deterministic=False):
        """
        obs: np.ndarray shape (obs_dim,)
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, logp = self.actor.sample(obs_t)
        if deterministic:
            return torch.tanh(self.actor.forward(obs_t)[0]).squeeze(0).cpu().numpy()
        return action.squeeze(0).cpu().numpy()

    # ------------------------------------------------
    def update(self, batch):
        """
        batch = (obs, act, rew, next_obs, done)
        """
        obs, act, rew, next_obs, done = batch

        # =========================
        # 1. Critic update (Soft Bellman)
        # =========================
        with torch.no_grad():
            next_action, next_logp = self.actor.sample(next_obs)
            q1_next = self.q1_target(next_obs, next_action)
            q2_next = self.q2_target(next_obs, next_action)
            q_next = torch.min(q1_next, q2_next)

            target_q = rew + self.gamma * (1 - done) * (q_next - self.alpha * next_logp)

        q1 = self.q1(obs, act)
        q2 = self.q2(obs, act)

        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        # =========================
        # 2. Actor update
        # =========================
        action_pi, logp_pi = self.actor.sample(obs)
        q1_pi = self.q1(obs, action_pi)
        q2_pi = self.q2(obs, action_pi)
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * logp_pi - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # =========================
        # 3. Alpha (entropy) update
        # =========================
        alpha_loss = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # =========================
        # 4. Soft update target critics
        # =========================
        self.soft_update(self.q1, self.q1_target)
        self.soft_update(self.q2, self.q2_target)

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item(),
        }

    # ------------------------------------------------
    def soft_update(self, net, net_target):
        for p, p_targ in zip(net.parameters(), net_target.parameters()):
            p_targ.data.mul_(1 - self.tau)
            p_targ.data.add_(self.tau * p.data)

    # ------------------------------------------------
    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }, path)

    # ------------------------------------------------
    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_target.load_state_dict(ckpt["q1_target"])
        self.q2_target.load_state_dict(ckpt["q2_target"])
        self.log_alpha.data.copy_(ckpt["log_alpha"].to(self.device))
