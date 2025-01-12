import math

import numpy as np

import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils.step_api_compatibility import step_api_compatibility

import Box2D
from Box2D.b2 import (
    circleShape,
    contactListener,
    edgeShape,
    fixtureDef,
    polygonShape,
    vec2,
    revoluteJointDef,
)

VIEWPORT_W = 600
VIEWPORT_H = 400
VIEWPORT_MIN = min(VIEWPORT_W, VIEWPORT_H)
TRANSFORM_VEC = lambda xy: vec2(
    (VIEWPORT_W - VIEWPORT_MIN) / 2 + VIEWPORT_MIN / 2 * (xy[0] + 1),
    (VIEWPORT_H - VIEWPORT_MIN) / 2 + VIEWPORT_MIN / 2 * (xy[1] + 1),
)
TRANSFORM_RADIUS = lambda r: r * VIEWPORT_MIN / 2


class ContactDetector(contactListener):
    def __init__(self):
        contactListener.__init__(self)
        self.rocket = None
        self.planets = []

    def BeginContact(self, contact):
        if self.rocket in [contact.fixtureA.body, contact.fixtureB.body]:
            print("Rocket hit the planet!")
            self.rocket.ApplyForceToCenter((0, 10), True)

    def EndContact(self, contact):
        pass


class Rocket(gym.Env):
    def __init__(self, rocket_pos_ang_size, planets_pos_rad, render_mode=None):
        self.rocket_pos_ang_size = rocket_pos_ang_size
        self.planets_pos_rad = planets_pos_rad
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        low = np.array(
            [
                -10.0,  # v_x
                -10.0,  # v_y
                -10.0,   # a_x
                -10.0,   # a_y
                -2 * math.pi,  # theta
                -10.0,  # theta_dot
                -10.0,  # theta_acc
            ] + [0] * len(self.planets_pos_rad)  # 0 is the min distance
        ).astype(np.float32)

        high = np.array(
            [
                10.0,  # v_x
                10.0,  # v_y
                10.0,   # a_x
                10.0,   # a_y
                2 * math.pi,  # theta
                10.0,  # theta_dot
                10.0,  # theta_acc
            ] + [4] * len(self.planets_pos_rad)  # 4 is the max distance
        ).astype(np.float32)

        self.observation_space = gym.spaces.Box(low, high)
        self.action_space = gym.spaces.Discrete(4)

    def reset(self):
        super().reset()

        # self.rocket_pos_ang_size = rocket_pos_ang_size
        # self.planets_pos = planets_pos

        self.world = Box2D.b2World()
        self.world.gravity = (0, 0)

        rocket_pos = self.rocket_pos_ang_size[:2]
        rocket_deg = self.rocket_pos_ang_size[2]
        rocket_size = self.rocket_pos_ang_size[3]
        rocket_shape = [
            (rocket_size, 0),
            (-rocket_size, 0),
            (0, 2.5*rocket_size)
        ]
        self.rocket = self.world.CreateDynamicBody(
            position=rocket_pos,
            angle=rocket_deg,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=rocket_shape)
            )
        )
        self.rocket.color1 = (128, 128, 128)
        self.rocket.color2 = (128, 128, 128)

        self.rocket.ApplyForceToCenter((0, 10), True)

        self.planets = []
        for planet_pos_rad in self.planets_pos_rad:
            planet = self.world.CreateStaticBody(
                position=planet_pos_rad[:2], angle=0.0,
                fixtures=fixtureDef(
                    shape=circleShape(radius=planet_pos_rad[2])
                )
            )
            planet.color1 = (255, 255, 255)
            planet.color2 = (255, 255, 255)
            self.planets.append(planet)

        self.world.contactListener = ContactDetector()
        self.world.contactListener.rocket = self.rocket
        self.world.contactListener.planets = self.planets

        self.drawlist = [self.rocket] + self.planets

        if self.render_mode == "human":
            self.render()

        return self.step(0)[0], {}

    def _get_observation(self):
        rocket_pos = self.rocket.position
        rocket_v = self.rocket.linearVelocity
        # rocket_a = self.rocket.linearAcceleration
        rocket_theta = self.rocket.angle
        rocket_theta_dot = self.rocket.angularVelocity
        # rocket_theta_acc = self.rocket.angularAcceleration

        planets_pos = [planet.position for planet in self.planets]

        return np.array(
            [
                rocket_v.x,
                rocket_v.y,
                # rocket_a.x,
                # rocket_a.y,
                rocket_theta,
                rocket_theta_dot,
                # rocket_theta_acc,
            ]
            + [np.linalg.norm(rocket_pos - planet_pos) for planet_pos in planets_pos]
        )

    def step(self, action):
        assert self.rocket is not None
        obs = self._get_observation()
        reward = 0
        terminated = False
        return obs, reward, terminated, False, {}

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
            self.clock.tick(self.metadata.get("reder_fps", 30))
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )


def demo_rocket(env, render=False):
    s, info = env.reset()
    steps = 0
    while True:
        s, r, terminated, truncated, info = step_api_compatibility(env.step(0), True)

        if render:
            still_open = env.render()
            if still_open is False:
                break

            steps += 1

        if terminated or truncated:
            break

    if render:
        env.close()


if __name__ == "__main__":
    env = Rocket(
        rocket_pos_ang_size=(0, 0, np.pi/4, 0.1),
        planets_pos_rad=[(-0.5, 0.5, 0.1), (0.5, -0.5, 0.1)],
        render_mode="human"
    )
    demo_rocket(env, render=True)
