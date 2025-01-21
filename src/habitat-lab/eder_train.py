import os
import numpy as np
import cv2
import imageio
from datetime import datetime

import habitat_sim
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions import Normal


########################################
# 1. Create Habitat Simulator Configuration
########################################
def make_cfg(scene_path, width=256, height=256, sensor_height=1.5):
    """
    Construct Habitat Simulator Configuration
    scene_path: Path to the scene file
    width, height: Sensor image resolution
    sensor_height: Sensor height
    """
    # Simulator config
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.enable_physics = False

    # Sensor configuration (using RGB only)
    sensor_specs = []
    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [height, width]
    color_sensor_spec.position = [0.0, sensor_height, 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_spec)

    # Agent configuration (defining discrete actions: move_forward, turn_left, turn_right)
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


########################################
# 2. Environment Helper Functions
########################################
def transform_rgb_bgr(image):
    """Habitat outputs RGB, convert to BGR for OpenCV display or processing"""
    # image.shape: (H, W, 3), RGB
    # Convert to BGR channels
    return image[..., [2, 1, 0]]

def calculate_distance(point1, point2):
    """Calculate Euclidean distance"""
    return np.linalg.norm(np.array(point1) - np.array(point2))

def generate_random_goal(sim):
    """Sample a random goal point in the navigable area"""
    pathfinder = sim.pathfinder
    random_goal = pathfinder.get_random_navigable_point()
    return random_goal


########################################
# 3. Neural Network: Visual Encoder
########################################
class VisualEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super(VisualEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),   # -> (32, 63, 63)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # -> (64, 30, 30)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # -> (64, 28, 28)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x shape: (N,3,H,W)
        return self.encoder(x)


########################################
# 4. DDPG Core Networks: Actor & Critic
########################################
class Actor(nn.Module):
    """
    Actor: Takes state features as input and outputs continuous actions (mean, std).
    Example action_dim=2 (e.g., (move_val, turn_val))
    """
    def __init__(self, feature_dim=256, action_dim=2):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.mean = nn.Linear(64, action_dim)
        self.log_std = nn.Linear(64, action_dim)

    def forward(self, state_embed):
        x = self.fc(state_embed)
        mean = self.mean(x)
        log_std = self.log_std(x)
        # Clamp std range
        std = torch.clamp(log_std, -20, 2).exp()
        return mean, std


class Critic(nn.Module):
    """
    Critic: Takes (state_embed, action) as input and outputs Q(s,a)
    """
    def __init__(self, feature_dim=256, action_dim=2):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state_embed, action):
        x = torch.cat([state_embed, action], dim=-1)
        return self.fc(x)


########################################
# 5. DDPG Agent Wrapper
########################################
class DDPGAgent:
    def __init__(self, state_dim=256, action_dim=2,
                 lr_actor=1e-4, lr_critic=1e-3, tau=0.005, gamma=0.99):
        self.gamma = gamma
        self.tau = tau

        # Actor & Critic
        self.actor = Actor(feature_dim=state_dim, action_dim=action_dim).cuda()
        self.critic = Critic(feature_dim=state_dim, action_dim=action_dim).cuda()

        # Target networks (for stable training)
        self.actor_target = Actor(feature_dim=state_dim, action_dim=action_dim).cuda()
        self.critic_target = Critic(feature_dim=state_dim, action_dim=action_dim).cuda()

        # Initialize target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def to(self, device):
        """Conveniently move all internal models to the specified device"""
        self.actor.to(device)
        self.critic.to(device)
        self.actor_target.to(device)
        self.critic_target.to(device)
        return self

    def choose_action(self, state_embed, noise_scale=0.1):
        """
        DDPG uses a deterministic policy: actor outputs mean. Add noise for exploration during training.
        state_embed: (1, feature_dim)
        """
        self.actor.eval()
        with torch.no_grad():
            mean, _ = self.actor(state_embed)
        self.actor.train()

        # Add Gaussian noise
        action = mean + noise_scale * torch.randn_like(mean).cuda()
        # Assume action range [-1,1]
        action = torch.clamp(action, -1.0, 1.0)
        return action

    def update(self, batch, encoder, device="cuda"):
        """
        Update Actor & Critic using a batch of (s, a, r, s', done)
        Each field in the batch is a list or tensor
        """
        # Concatenate into tensors
        states = torch.stack(batch["states"]).float().to(device)       # shape (B, 3, H, W)
        actions = torch.stack(batch["actions"]).float().to(device)     # shape (B, action_dim)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float).to(device)  # (B,)
        next_states = torch.stack(batch["next_states"]).float().to(device)      # (B, 3, H, W)
        dones = torch.tensor(batch["dones"], dtype=torch.float).to(device)      # (B,)

        # Encode features
        with torch.no_grad():
            state_embeds = encoder(states)            # (B, feature_dim)
            next_state_embeds = encoder(next_states)  # (B, feature_dim)

        # Critic update
        with torch.no_grad():
            # Target Actor generates next actions
            next_actions_mean, _ = self.actor_target(next_state_embeds)
            # Compute target Q
            target_Q = self.critic_target(next_state_embeds, next_actions_mean).squeeze(-1)
            target_value = rewards + self.gamma * (1 - dones) * target_Q

        current_Q = self.critic(state_embeds, actions).squeeze(-1)
        critic_loss = F.mse_loss(current_Q, target_value)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor update
        pred_actions_mean, _ = self.actor(state_embeds)
        actor_loss = -self.critic(state_embeds, pred_actions_mean).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Soft update
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item()


########################################
# 6. EDER Diverse Experience Replay Buffer (Based on Low-Dimensional Features)
########################################
class EDERReplayBuffer:
    """
    In this version, we call the Encoder to obtain low-dimensional features during sampling,
    then perform DPP on these features. This way, the matrix size is only related to embed_dim (e.g., 256x256),
    avoiding high memory usage from raw pixels. Also, ensure images are permuted (2,0,1) to meet Conv2d input requirements.
    """
    def __init__(self, max_size=10000, segment_len=2, device="cuda"):
        self.max_size = max_size
        self.segment_len = segment_len
        self.device = device

        self.buffer = []
        self.ptr = 0

    def store(self, transition):
        """
        transition: (state, action, reward, next_state, done)
        state/next_state: shape (H,W,3) numpy (BGR or RGB is fine, as long as consistent)
        """
        if len(self.buffer) < self.max_size:
            self.buffer.append(transition)
        else:
            self.buffer[self.ptr] = transition
            self.ptr = (self.ptr + 1) % self.max_size

    def compute_diversity(self, embed_batch):
        """
        compute_diversity is no longer based on raw pixels, but on the embeddings of each state using determinants.
        embed_batch: shape = (segment_len, embed_dim)
        """
        # L2 normalization
        norms = embed_batch.norm(dim=1, keepdim=True) + 1e-6
        normed_states = embed_batch / norms  # (segment_len, embed_dim)

        # Compute L = M^T M
        M = normed_states.t()             # (embed_dim, segment_len)
        L = M @ M.t()                     # (embed_dim, embed_dim)

        # Cholesky decomposition
        try:
            L_c = torch.linalg.cholesky(L)
            det_val = torch.prod(torch.diagonal(L_c))**2
        except:
            det_val = torch.tensor(1e-8, device=self.device)

        return det_val.item()

    def sample_eder_batch(self, encoder, batch_size=64):
        """
        First segment the buffer -> use Encoder to get embeddings -> compute diversity (determinant) ->
        reject samples -> return a batch for training
        """
        if len(self.buffer) < self.segment_len:
            return None

        # Construct all sub-trajectories
        segments = []
        for i in range(0, len(self.buffer) - self.segment_len + 1, self.segment_len):
            subtraj = self.buffer[i : i + self.segment_len]
            segments.append(subtraj)

        if len(segments) == 0:
            return None

        diversities = []
        for subtraj in segments:
            # Encode each state in the trajectory using the Encoder -> embed_stack
            embed_stack = []
            for (s, a, r, s_next, d) in subtraj:
                # s shape (H, W, 3) -> (1, 3, H, W) float
                s_tensor = torch.from_numpy(s).float().permute(2,0,1).unsqueeze(0).to(self.device)
                s_tensor /= 255.0
                with torch.no_grad():
                    s_embed = encoder(s_tensor)  # (1, embed_dim)
                embed_stack.append(s_embed)

            # Concatenate into (segment_len, embed_dim)
            embed_stack = torch.cat(embed_stack, dim=0)  # (b, embed_dim)

            # Compute determinant
            diversity_score = self.compute_diversity(embed_stack)
            diversities.append(diversity_score)

        diversities = np.array(diversities)
        max_div = np.max(diversities) if len(diversities) > 0 else 1e-8

        # Rejection sampling: only keep a certain number of high-diversity sub-trajectories
        accepted_segments = []
        for idx, d_score in enumerate(diversities):
            u = np.random.rand()
            if max_div > 1e-12:
                if u <= (d_score / max_div):
                    accepted_segments.append(segments[idx])
            else:
                # If max_div is very small, accept all by default
                accepted_segments.append(segments[idx])

        if len(accepted_segments) == 0:
            return None

        # Randomly sample batch_size transitions from accepted segments
        all_transitions = []
        for seg in accepted_segments:
            all_transitions.extend(seg)

        np.random.shuffle(all_transitions)
        sampled_transitions = (
            all_transitions[:batch_size]
            if len(all_transitions) > batch_size
            else all_transitions
        )

        # Construct training batch
        batch = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": []
        }
        for (s, a, r, s_next, d) in sampled_transitions:
            # state
            s_t = torch.from_numpy(s).float().permute(2,0,1) / 255.0   # (3,H,W)
            # next_state
            ns_t = torch.from_numpy(s_next).float().permute(2,0,1) / 255.0
            batch["states"].append(s_t)
            batch["actions"].append(torch.from_numpy(a))
            batch["rewards"].append(r)
            batch["next_states"].append(ns_t)
            batch["dones"].append(d)

        return batch


########################################
# 7. Main Training Loop: DDPG + EDER
########################################
def train_ddpg_eder():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -- Scene path (modify to your local Habitat data path) --
    data_path = "/home/xxx/habitat-lab/"
    test_scene = os.path.join(data_path, "data/hm3d/hm3d-val-habitat-v0.2/00800-TEEsavR23oF/TEEsavR23oF.basis.glb")

    # -- Create Habitat Simulator --
    config = make_cfg(test_scene)
    sim = habitat_sim.Simulator(config)
    agent = sim.initialize_agent(0)

    # Initialize Agent position
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array([0.0, 1.0, 0.0])
    agent.set_state(agent_state)

    # Random goal
    goal_pos = generate_random_goal(sim)
    print("Random Goal Position: ", goal_pos)

    # -- DDPG + EDER --
    encoder = VisualEncoder(output_dim=256).to(device)
    ddpg_agent = DDPGAgent(state_dim=256, action_dim=2).to(device)
    eder_buffer = EDERReplayBuffer(max_size=20000, segment_len=2, device=device)

    max_episodes = 100
    max_steps_per_episode = 200
    global_step = 0

    log_file = "ddpg_eder_log_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt"

    for ep in range(max_episodes):
        # Reset agent position
        agent_state.position = np.array([0.0, 1.0, 0.0])
        agent.set_state(agent_state)

        done = False
        total_reward = 0
        steps = 0

        # Record video
        #video_writer = imageio.get_writer(f'episode_{ep}_video.mp4', fps=20)

        for step_i in range(max_steps_per_episode):
            global_step += 1
            steps += 1

            # Current frame
            obs = sim.get_sensor_observations()
            rgb = obs["color_sensor"]  # shape (H,W,3) (RGB)
            rgb_bgr = transform_rgb_bgr(rgb)  # Convert to BGR (H,W,3)

            # Convert to tensor and send to encoder
            state_tensor = torch.from_numpy(rgb_bgr).float().permute(2,0,1).unsqueeze(0).to(device)
            state_tensor /= 255.0  # Normalize
            with torch.no_grad():
                state_embed = encoder(state_tensor)  # (1, embed_dim)

            # Choose action (move_val, turn_val)
            action = ddpg_agent.choose_action(state_embed, noise_scale=0.2)
            action_np = action.squeeze().detach().cpu().numpy()
            move_val, turn_val = action_np[0], action_np[1]

            # Simple mapping: if move_val > 0, move forward; otherwise, turn based on turn_val sign
            if move_val > 0:
                sim.step("move_forward")
            else:
                if turn_val > 0:
                    sim.step("turn_right")
                else:
                    sim.step("turn_left")

            # Next state
            next_obs = sim.get_sensor_observations()
            next_rgb = next_obs["color_sensor"]
            next_rgb_bgr = transform_rgb_bgr(next_rgb)

            # Extrinsic reward
            current_pos = agent.get_state().position
            dist2goal = calculate_distance(current_pos, goal_pos)
            reward = -dist2goal
            done = False

            if dist2goal < 0.5:
                reward += 100
                done = True

            if step_i == max_steps_per_episode - 1:
                reward -= 50
                done = True

            total_reward += reward

            # Store in EDER
            transition = (
                rgb_bgr,           # state (H,W,3) numpy
                action_np,         # action (2,)
                reward,            # float
                next_rgb_bgr,      # next_state (H,W,3) numpy
                float(done)
            )
            eder_buffer.store(transition)

            # Visualization
            cv2.imshow("AgentView", next_rgb_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Training interrupted by user!")
                sim.close()
                return

            # Save video frame
            # video_writer.append_data(next_rgb_bgr)

            if done:
                break

            # Periodic updates
            if global_step % 50 == 0:
                batch = eder_buffer.sample_eder_batch(encoder, batch_size=64)
                if batch is not None:
                    critic_loss, actor_loss = ddpg_agent.update(batch, encoder, device)
                    # To train the encoder end-to-end, remove the with torch.no_grad() and related steps
                    # Here we demonstrate the most basic structure

        # End of episode
        # video_writer.close()

        log_str = f"Episode {ep} | Steps: {steps} | Reward: {total_reward:.2f} | Dist2Goal: {dist2goal:.2f}"
        print(log_str)
        with open(log_file, "a") as f:
            f.write(log_str + "\n")

    sim.close()
    cv2.destroyAllWindows()


########################################
# 8. Main Entry
########################################
if __name__ == "__main__":
    train_ddpg_eder()
