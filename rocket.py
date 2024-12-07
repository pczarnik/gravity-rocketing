import math

import numpy as np

import gymnasium as gym
from gymnasium.error import DependencyNotInstalled

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
        self.rocket_pos = rocket_pos

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

    def reset(self, rocket_pos, planets_pos):
        self.planets_pos = planets_pos
        self.rocket_pos = rocket_pos

        self.world = Box2D.b2World()
        self.world.gravity = (0, 0)

        self.rocket = self.world.CreateDynamicBody(
            position=self.rocket_pos,
            angle=0.0,
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

        return self._get_observation()

    def _get_observation(self):
        rocket_pos = self.rocket.position
        rocket_v = self.rocket.linearVelocity
        rocket_a = self.rocket.linearAcceleration
        rocket_theta = self.rocket.angle
        rocket_theta_dot = self.rocket.angularVelocity
        rocket_theta_acc = self.rocket.angularAcceleration

        planets_pos = [planet.position for planet in self.planets]

        return np.array(
            [
                rocket_v.x,
                rocket_v.y,
                rocket_a.x,
                rocket_a.y,
                rocket_theta,
                rocket_theta_dot,
                rocket_theta_acc,
            ]
            + [np.linalg.norm(rocket_pos - planet_pos) for planet_pos in planets_pos]
        )
    
    def step(self, action):
        assert self.lander is not None

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

        pygame.transform.scale(self.surf, (SCALE, SCALE))
        pygame.draw.rect(self.surf, (255, 255, 255), self.surf.get_rect())

        for obj in self.particles:
            obj.ttl -= 0.15
            obj.color1 = (
                int(max(0.2, 0.15 + obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
            )
            obj.color2 = (
                int(max(0.2, 0.15 + obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
            )

        self._clean_particles(False)

        for p in self.sky_polys:
            scaled_poly = []
            for coord in p:
                scaled_poly.append((coord[0] * SCALE, coord[1] * SCALE))
            pygame.draw.polygon(self.surf, (0, 0, 0), scaled_poly)
            gfxdraw.aapolygon(self.surf, scaled_poly, (0, 0, 0))

        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color1,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color2,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )

                else:
                    path = [trans * v * SCALE for v in f.shape.vertices]
                    pygame.draw.polygon(self.surf, color=obj.color1, points=path)
                    gfxdraw.aapolygon(self.surf, path, obj.color1)
                    pygame.draw.aalines(
                        self.surf, color=obj.color2, points=path, closed=True
                    )

                for x in [self.helipad_x1, self.helipad_x2]:
                    x = x * SCALE
                    flagy1 = self.helipad_y * SCALE
                    flagy2 = flagy1 + 50
                    pygame.draw.line(
                        self.surf,
                        color=(255, 255, 255),
                        start_pos=(x, flagy1),
                        end_pos=(x, flagy2),
                        width=1,
                    )
                    pygame.draw.polygon(
                        self.surf,
                        color=(204, 204, 0),
                        points=[
                            (x, flagy2),
                            (x, flagy2 - 10),
                            (x + 25, flagy2 - 5),
                        ],
                    )
                    gfxdraw.aapolygon(
                        self.surf,
                        [(x, flagy2), (x, flagy2 - 10), (x + 25, flagy2 - 5)],
                        (204, 204, 0),
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

if __name__ == "__main__":
    rocket = Rocket(rocket_pos=(0, 0), planets_pos=[(0, 1), (1, 0)])
    rocket.reset(rocket_pos=(0, 0), planets_pos=[(0, 1), (1, 0)])
    rocket.step(0)