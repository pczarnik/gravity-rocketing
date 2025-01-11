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
    revoluteJointDef,
)

VIEWPORT_W = 600
VIEWPORT_H = 400


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
    def __init__(self, rocket_pos, planets_pos):
        self.planets_pos = planets_pos
        self.rocket_pos_deg = rocket_pos
        self.render_mode = "human"
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
            ] + [0] * len(self.planets_pos)  # 0 is the min distance
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
            ] + [4] * len(self.planets_pos)  # 4 is the max distance
        ).astype(np.float32)

        self.observation_space = gym.spaces.Box(low, high)
        self.action_space = gym.spaces.Discrete(4)

    def reset(self, rocket_pos_deg=(0, 0, 0), planets_pos=[(-0.5, 0), (0.5, 0)]):
        self.rocket_pos_deg = rocket_pos_deg
        self.planets_pos = planets_pos

        self.world = Box2D.b2World()
        self.world.gravity = (0, 0)

        self.rocket = self.world.CreateDynamicBody(
            position=self.rocket_pos_deg[:2],
            angle=self.rocket_pos_deg[2],
            fixtures=fixtureDef(
                shape=polygonShape(box=(0.1, 0.1))
            )
        )

        self.rocket.ApplyForceToCenter((0, 10), True)

        self.planets = []
        for planet_pos in self.planets_pos:
            planet = self.world.CreateStaticBody(
                position=planet_pos, angle=0.0,
                fixtures=fixtureDef(
                    shape=circleShape(radius=0.1)
                )
            )
            self.planets.append(planet)

        self.world.contactListener = ContactDetector()
        self.world.contactListener.rocket = self.rocket
        self.world.contactListener.planets = self.planets

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

        for planet in self.planets:
            planet_pos = planet.position
            planet_pos_screen = (
                int(VIEWPORT_W / 2 + planet_pos[0] * VIEWPORT_W / 2),
                int(VIEWPORT_H / 2 - planet_pos[1] * VIEWPORT_H / 2),
            )
            pygame.draw.circle(
                self.surf,
                (0, 0, 255),
                planet_pos_screen,
                int(0.1 * VIEWPORT_W / 2),
            )
            gfxdraw.aacircle(
                self.surf,
                planet_pos_screen[0],
                planet_pos_screen[1],
                int(0.1 * VIEWPORT_W / 2),
                (0, 0, 255),
            )

        rocket_pos = self.rocket.position
        rocket_pos_screen = (
            int(VIEWPORT_W / 2 + rocket_pos[0] * VIEWPORT_W / 2),
            int(VIEWPORT_H / 2 - rocket_pos[1] * VIEWPORT_H / 2),
        )
        pygame.draw.polygon(
            self.surf,
            (255, 0, 0),
            [
            (rocket_pos_screen[0] + int(0.1 * VIEWPORT_W / 2), rocket_pos_screen[1]),
            (rocket_pos_screen[0] - int(0.1 * VIEWPORT_W / 2), rocket_pos_screen[1] + int(0.1 * VIEWPORT_H / 2)),
            (rocket_pos_screen[0] - int(0.1 * VIEWPORT_W / 2), rocket_pos_screen[1] - int(0.1 * VIEWPORT_H / 2)),
            ],
        )
        gfxdraw.aapolygon(
            self.surf,
            [
            (rocket_pos_screen[0] + int(0.1 * VIEWPORT_W / 2), rocket_pos_screen[1]),
            (rocket_pos_screen[0] - int(0.1 * VIEWPORT_W / 2), rocket_pos_screen[1] + int(0.1 * VIEWPORT_H / 2)),
            (rocket_pos_screen[0] - int(0.1 * VIEWPORT_W / 2), rocket_pos_screen[1] - int(0.1 * VIEWPORT_H / 2)),
            ],
            (255, 0, 0),
        )

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
    env = Rocket(rocket_pos=(0, 0), planets_pos=[(0, 1), (1, 0)])
    demo_rocket(env, render=True)
