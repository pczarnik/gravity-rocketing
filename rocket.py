import math

import numpy as np

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

FPS = 30
SCALE = 1.75
VIEWPORT_W = 600
VIEWPORT_H = 400
VIEWPORT_MIN = min(VIEWPORT_W, VIEWPORT_H) / SCALE
TRANSFORM_VEC = lambda xy: vec2(
    (VIEWPORT_W - VIEWPORT_MIN) / 2  + (VIEWPORT_MIN / 2 * (xy[0] + 1)),
    (VIEWPORT_H - VIEWPORT_MIN) / 2 + (VIEWPORT_MIN / 2 * (xy[1] + 1)),
)
TRANSFORM_RADIUS = lambda r: r * VIEWPORT_MIN / 2
GRAVITY_CONST = 1e-3
PLANET_DENS = 10
EPS = 1e-6
PULSE_POWER = 2e-4
ROTATION_POWER = 5e-8
RAD2DEG = lambda x: x * 180 / np.pi


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

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(self, rocket_pos_ang_size_mass, planets_pos_mass, render_mode=None):
        self.rocket_pos_ang_size_mass = np.array(rocket_pos_ang_size_mass, dtype=np.float64)
        self.planets_pos_mass = np.array(planets_pos_mass, dtype=np.float64)

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.world = None

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
                    shape=circleShape(radius=planet_pos_mass[2] / PLANET_DENS)
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
            force = PULSE_POWER * unit_v
            self.rocket.ApplyForceToCenter(force, True)

        if action == 2:
            self.rocket.ApplyAngularImpulse(
                ROTATION_POWER,
                True,
            )
        elif action == 3:
            self.rocket.ApplyAngularImpulse(
                -ROTATION_POWER,
                True,
            )

        acc = vec2(0, 0)
        for (x, y, mass) in self.planets_pos_mass:
            delta = vec2(x, y) - self.rocket.position
            dist = np.linalg.norm(delta)

            acc += delta * mass / (dist ** 3 + EPS)

        force = acc * GRAVITY_CONST * self.rocket_mass
        self.rocket.ApplyForceToCenter(force, True)

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.rocket.position
        vel = self.rocket.linearVelocity

        if max(np.abs(pos)) > 2:
            self.game_over = True

        state = [
            pos.x,
            pos.y,
            vel.x / FPS,
            vel.y / FPS,
            self.rocket.angle,
            self.rocket.angularVelocity / FPS,
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
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))

        self.surf.fill((0, 0, 0))

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color1,
                        center=TRANSFORM_VEC(trans * f.shape.pos),
                        radius=TRANSFORM_RADIUS(f.shape.radius),
                    )
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color2,
                        center=TRANSFORM_VEC(trans * f.shape.pos),
                        radius=TRANSFORM_RADIUS(f.shape.radius),
                    )

                else:
                    path = [TRANSFORM_VEC(trans * v) for v in f.shape.vertices]
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
    vec2planet = env.planets_pos_mass[0][:2] - s[:2]
    angle2planet = np.atan2(vec2planet[1], vec2planet[0]) + 3/2*np.pi
    rocket_angle = s[4]
    relative_angle = (angle2planet - rocket_angle) % (2*np.pi)

    rocket_speed_dir = np.atan2(s[3], s[2]) + 3/2*np.pi
    relative_speed_angle = (rocket_speed_dir - angle2planet) % (2*np.pi)

    speed = np.linalg.norm(s[2:4])
    rot_speed = s[5]

    relative_angle = relative_angle / np.pi
    relative_speed_angle = relative_speed_angle / np.pi

    relative_angle = (1 - relative_angle) % 2 - 1
    relative_speed_angle = (1 - relative_speed_angle) % 2 - 1

    action = 0
    if relative_angle < -0.1 and rot_speed <= 0.04:
        print(1)
        action = 2
    elif relative_angle > 0.1 and rot_speed >= -0.04:
        print(2)
        action = 3

    if relative_angle < -0.1 and abs(relative_speed_angle) > 0.4 and rot_speed <= 0.07:
        print(3)
        action = 2
    elif relative_angle > 0.1 and abs(relative_speed_angle) > 0.4 and rot_speed >= -0.07:
        print(4)
        action = 3

    if abs(relative_angle) < 0.1 and abs(relative_speed_angle) > 0.7:
        print(5)
        action = 1

    if abs(relative_angle) < 0.01 and speed < 0.01:
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


if __name__ == "__main__":
    init_rocket_pos_ang = [-0.8, -0.8, np.pi/2 + 0.2]
    # init_rocket_pos_ang = np.random.uniform([-1.5, -1.5, -np.pi], [0, 0, np.pi])

    env = Rocket(
        rocket_pos_ang_size_mass=(*init_rocket_pos_ang, 0.03, 0.1),
        planets_pos_mass=[
            (0.5, 0.5, 1),
            # (-0.5, 0.5, 2),
            (0.75, -0.75, 0.5),
        ],
        render_mode="human"
    )
    demo_rocket(env, render=True)
