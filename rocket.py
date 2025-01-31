import math

import numpy as np

from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils.step_api_compatibility import step_api_compatibility

import Box2D
from Box2D.b2 import (
    circleShape,
    contactListener,
    fixtureDef,
    polygonShape,
    vec2,
)


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if self.env.rocket in [contact.fixtureA.body, contact.fixtureB.body]:
            if self.env.destination in [contact.fixtureA.body, contact.fixtureB.body]:
                print("Rocket landed!")
                self.env.landed = True
            else:
                print("Rocket hit the planet!")
                self.env.game_over = True

    def EndContact(self, contact):
        pass


class Rocket(gym.Env):

    def __init__(self, rocket_pos_ang_size_mass, planets_pos_mass, render_mode=None, config=None):
        self.rocket_pos_ang_size_mass = np.array(rocket_pos_ang_size_mass, dtype=np.float64)
        self.planets_pos_mass = np.array(planets_pos_mass, dtype=np.float64)

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.world = None
        self.config = config if config else {}

        env_config = self.config.get('environment', {})

        self.fps = int(env_config.get('fps', 30))
        self.scale = float(env_config.get('scale', 2.5))
        self.bound = float(env_config.get('bound', 1.1 * self.scale))
        self.viewport_w = int(env_config.get('viewport_w', 1200))
        self.viewport_h = int(env_config.get('viewport_h', 800))
        self.gravity_const = float(env_config.get('gravity_const', 1e-3))
        self.planet_dens = float(env_config.get('planet_dens', 10))
        self.eps = float(env_config.get('eps', 1e-6))
        self.pulse_power = float(env_config.get('pulse_power', 2e-4))
        self.rotation_power = float(env_config.get('rotation_power', 5e-8))

        self.viewport_min = min(self.viewport_w, self.viewport_h) / self.scale
        self.transform_vec = lambda xy: vec2(
            (self.viewport_w - self.viewport_min) / 2  + (self.viewport_min / 2 * (xy[0] + 1)),
            (self.viewport_h - self.viewport_min) / 2 + (self.viewport_min / 2 * (xy[1] + 1)),
        )
        self.transform_radius = lambda r: r * self.viewport_min / 2

        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": self.fps,
        }

        low = np.array(
            [
                -10.0,  # v_x
                -10.0,  # v_y
                -2 * math.pi,  # theta
                -10.0,  # theta_dot
            ] + [0] * len(self.planets_pos_mass)  # 0 is the min distance
        ).astype(np.float32)

        high = np.array(
            [
                10.0,  # v_x
                10.0,  # v_y
                2 * math.pi,  # theta
                10.0,  # theta_dot
            ] + [4] * len(self.planets_pos_mass)  # 4 is the max distance
        ).astype(np.float32)

        self.observation_space = gym.spaces.Box(low, high)

        # 0 - do nothing
        # 1 - forward impulse
        # 2 - left rotation impulse
        # 3 - right rotation impulse
        self.action_space = gym.spaces.Discrete(4)

    def _destroy(self):
        if self.world is None:
            return
        self.world.contactListener = None
        self.world.DestroyBody(self.rocket)
        self.rocket = None
        for planet in self.planets:
            self.world.DestroyBody(planet)
        self.planets = []
        self.world = None

    def reset(self, *, seed=None):
        super().reset(seed=seed)
        self._destroy()

        self.world = Box2D.b2World(gravity=(0, 0))
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.landed = False
        self.game_over = False

        rocket_pos = self.rocket_pos_ang_size_mass[:2]
        rocket_deg = self.rocket_pos_ang_size_mass[2]
        rocket_size = self.rocket_pos_ang_size_mass[3]
        self.rocket_mass = self.rocket_pos_ang_size_mass[4]
        rocket_shape = [
            (rocket_size, 0),
            (-rocket_size, 0),
            (0, 2.5*rocket_size)
        ]
        self.rocket = self.world.CreateDynamicBody(
            position=rocket_pos,
            angle=rocket_deg,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=rocket_shape),
                density=self.rocket_mass,
            )
        )
        self.rocket.color1 = (128, 128, 128)
        self.rocket.color2 = (128, 128, 128)

        self.planets = []
        for planet_pos_mass in self.planets_pos_mass:
            planet = self.world.CreateStaticBody(
                position=planet_pos_mass[:2], angle=0.0,
                fixtures=fixtureDef(
                    shape=circleShape(radius=planet_pos_mass[2] / self.planet_dens)
                )
            )
            planet.color1 = (255, 255, 255)
            planet.color2 = (255, 255, 255)
            self.planets.append(planet)

        self.destination = self.planets[0]

        self.drawlist = [self.rocket] + self.planets

        if self.render_mode == "human":
            self.render()

        return self.step(0)[0], {}

    def step(self, action):
        assert self.rocket is not None
        assert self.action_space.contains(action)

        if action == 1:
            unit_v = vec2(-np.sin(self.rocket.angle), np.cos(self.rocket.angle))
            force = self.pulse_power * unit_v
            self.rocket.ApplyForceToCenter(force, True)

        if action == 2:
            self.rocket.ApplyAngularImpulse(
                self.rotation_power,
                True,
            )
        elif action == 3:
            self.rocket.ApplyAngularImpulse(
                -self.rotation_power,
                True,
            )

        acc = vec2(0, 0)
        for (x, y, mass) in self.planets_pos_mass:
            delta = vec2(x, y) - self.rocket.position
            dist = np.linalg.norm(delta)

            acc += delta * mass / (dist ** 3 + self.eps)

        force = acc * self.gravity_const * self.rocket_mass
        self.rocket.ApplyForceToCenter(force, True)

        self.world.Step(1.0 / self.fps, 6 * 30, 2 * 30)

        pos = self.rocket.position
        vel = self.rocket.linearVelocity

        if max(np.abs(pos)) > self.bound:
            self.game_over = True

        state = [
            pos.x,
            pos.y,
            vel.x / self.fps,
            vel.y / self.fps,
            self.rocket.angle,
            self.rocket.angularVelocity / self.fps,
        ] + [
            np.linalg.norm(vec2(x, y) - pos)
            for (x, y, _) in self.planets_pos_mass
        ]

        reward = 0
        terminated = False
        if self.landed:
            reward += 100
            terminated = True

        if self.game_over:
            reward += -100
            terminated = True

        destination_pos = vec2(*self.planets_pos_mass[0][:2])
        destination_dist = np.linalg.norm(destination_pos - pos)
        reward += 10 * (4 - destination_dist)

        if self.render_mode == "human":
            self.render()

        return np.array(state, dtype=np.float32), reward, terminated, False, {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[box2d]"`'
            ) from e

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.viewport_w, self.viewport_h))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.viewport_w, self.viewport_h))

        self.surf.fill((0, 0, 0))

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color1,
                        center=self.transform_vec(trans * f.shape.pos),
                        radius=self.transform_radius(f.shape.radius),
                    )
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color2,
                        center=self.transform_vec(trans * f.shape.pos),
                        radius=self.transform_radius(f.shape.radius),
                    )

                else:
                    path = [self.transform_vec(trans * v) for v in f.shape.vertices]
                    pygame.draw.polygon(self.surf, color=obj.color1, points=path)
                    gfxdraw.aapolygon(self.surf, path, obj.color1)
                    pygame.draw.aalines(
                        self.surf, color=obj.color2, points=path, closed=True
                    )

        self.surf = pygame.transform.flip(self.surf, False, True)

        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


def heuristic(env, s):
    conf = env.config
    relative_angle_threshold = float(conf.get('relative_angle_threshold', 0.1))
    target_rot_speed_threshold = float(conf.get('target_rot_speed_threshold', 0.04))
    danger_angle_threshold = float(conf.get('danger_angle_threshold', 0.04))
    emergency_rot_speed_threshold = float(conf.get('emergency_rot_speed_threshold', 0.07))
    target_relative_angle = float(conf.get('target_relative_angle', 0.01))
    danger_pull_angle_threshold = float(conf.get('danger_pull_angle_threshold', 0.7))
    target_speed = float(conf.get('target_speed', 0.01))

    vec2planet = env.planets_pos_mass[0][:2] - s[:2]
    angle2planet = np.atan2(vec2planet[1], vec2planet[0]) / np.pi + 3/2
    rocket_angle = s[4] / np.pi
    relative_angle = (angle2planet - rocket_angle) % 2

    rocket_speed_dir = np.atan2(s[3], s[2]) / np.pi + 3/2
    relative_speed_angle = (rocket_speed_dir - angle2planet) % 2

    speed = np.linalg.norm(s[2:4])
    rot_speed = s[5]

    relative_angle = (1 - relative_angle) % 2 - 1
    relative_speed_angle = (1 - relative_speed_angle) % 2 - 1

    action = 0
    if relative_angle < -relative_angle_threshold and rot_speed <= target_rot_speed_threshold:
        print(1)
        action = 2
    elif relative_angle > relative_angle_threshold and rot_speed >= -target_rot_speed_threshold:
        print(2)
        action = 3

    if relative_angle < -relative_angle_threshold and abs(relative_speed_angle) > danger_angle_threshold and rot_speed <= emergency_rot_speed_threshold:
        print(3)
        action = 2
    elif relative_angle > relative_angle_threshold and abs(relative_speed_angle) > danger_angle_threshold and rot_speed >= -emergency_rot_speed_threshold:
        print(4)
        action = 3

    if abs(relative_angle) < relative_angle_threshold and abs(relative_speed_angle) > danger_pull_angle_threshold:
        print(5)
        action = 1

    if abs(relative_angle) < target_relative_angle and speed < target_speed:
        print(6)
        action = 1

    return action


def demo_rocket(env, render=False):
    total_reward = 0
    steps = 0
    s, info = env.reset()
    while True:
        a = heuristic(env, s)
        s, r, terminated, truncated, info = step_api_compatibility(env.step(a), True)
        total_reward += r

        if render:
            still_open = env.render()
            if still_open is False:
                break

        # if steps % 5 == 0 or terminated or truncated:
            # print("observations:", " ".join([f"{x:+0.2f}" for x in s]))
            # print(f"step {steps} reward {r:+0.2f} total_reward {total_reward:+0.2f}")
        steps += 1
        if terminated or truncated:
            break
    if render:
        env.close()
    return total_reward


def read_config(filename):
    with open(filename, 'r') as file:
        config = load(file, Loader=Loader)
    return config

if __name__ == "__main__":
    config = read_config('config.yaml')
    rocket_config = config.get('rocket', {})
    rocket_loc = rocket_config.get('loc', [-0.25, -0.75, 0])
    rocket_scale = rocket_config.get('scale', [0.25, 0.25, 1.047])
    init_rocket_pos_ang = np.random.normal(rocket_loc, rocket_scale)

    planets_config = config.get('planets', [
        [0.5, 0.5, 1],
        [0.75, -0.75, 0.5],
    ])

    env = Rocket(
        rocket_pos_ang_size_mass=(*init_rocket_pos_ang, 0.03, 0.1),
        planets_pos_mass=planets_config,
        render_mode="human",
        config=config
    )
    demo_rocket(env, render=True)
