#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Header
from scipy.spatial import KDTree
from std_msgs.msg import Int32
from copy import deepcopy

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200  # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb,queue_size=1)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb,queue_size=1)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        # rospy.Subscriber('/obstacle_waypoint', Obstacle, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher(
            'final_waypoints', Lane, queue_size=1)

        self.pose = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.base_lane = None
        self.stopline_wp_idx = -1

        self.loop()

    def loop(self):
        # Control the publishing frequency (better than rospy.spin())
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            # If init
            if self.pose and self.base_lane:
                self.publish_waypoints()
            rate.sleep()

    def get_closest_waypoint(self):
        closest_distance = float('inf')
        closest_wp_idx = 0
        for idx, waypoint in enumerate(self.base_waypoints):
            distance = self.straight_dist(
                waypoint.pose.pose.position,
                self.pose.pose.position)
            if distance < closest_distance:
                closest_wp_idx = idx
                closest_distance = distance
        curr_yaw = self.get_yaw(self.pose.pose.orientation)

        map_x = self.base_waypoints[closest_wp_idx].pose.pose.position.x
        map_y = self.base_waypoints[closest_wp_idx].pose.pose.position.y
        heading = math.atan2(
            (map_y - self.pose.pose.position.y),
            (map_x - self.pose.pose.position.x)
        )

        if abs(curr_yaw - heading) > math.pi / 4:
            return closest_wp_idx + 1
        else:
            return closest_wp_idx

    def get_yaw(self, orientation):
        _, _, yaw = tf.transformations.euler_from_quaternion(
            [
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            ]
        )
        return yaw

    def publish_waypoints(self):
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)

    def generate_lane(self):
        self.closest_wp_idx = self.get_closest_waypoint()

        waypoints = deepcopy(self.base_waypoints[
            self.closest_wp_idx:self.closest_wp_idx+LOOKAHEAD_WPS
        ])
        if self.next_stop_line_idx != -1:
            waypoints = self.decelerate_waypoints(waypoints,self.closest_wp_idx)



        lane = Lane()
        lane.waypoints = waypoints
        return lane




        

        # closest_wp_idx = self.get_closest_waypoint()
        # farthest_idx = closest_wp_idx + LOOKAHEAD_WPS
        # base_waypoints = self.base_lane.waypoints[closest_wp_idx:farthest_idx]

        # if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
        #     # Publish directly
        #     lane.waypoints = base_waypoints
        # else:
        #     # Calculate the decelerated waypoints
        #     lane.waypoints = self.decelerate_waypoints(
        #         base_waypoints, closest_wp_idx)

        # return lane

    def decelerate_waypoints(self, waypoints, closest_wp_idx):
        last_idx = self.next_stop_line_idx - self.next_wp_idx - 3
        last = waypoints[last_idx]
        last.twist.twist.linear.x = 0.
        for wp in waypoints[:last_idx][::-1]:
            dist = self.straight_dist(
                wp.pose.pose.position, last.pose.pose.position)
            vel = math.sqrt(2 * MAX_DECEL * dist)
            if vel < 1.:
                vel = 0.
            self.set_waypoint_velocity(
                wp, min(vel, self.get_waypoint_velocity(wp)))
        return waypoints

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints.waypoints

    # def waypoints_cb(self, waypoints):
    #     # Base waypoints
    #     self.base_lane = waypoints
    #     if not self.waypoints_2d:
    #         self.waypoints_2d = [
    #             [w.pose.pose.position.x, w.pose.pose.position.y] for w in waypoints.waypoints]
    #         self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.stopline_wp_idx=msg.data


    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        def dl(a, b): return math.sqrt(
            (a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position,
                       waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
