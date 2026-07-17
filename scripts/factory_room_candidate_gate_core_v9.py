#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pure state decisions for alternate continuous-scan candidate arrivals."""


def candidate_gate_decision(phase, approach_distance, final_distance,
                            travelled_distance, approach_tolerance,
                            final_tolerance, minimum_travel):
    """Return the next action without depending on ROS or wall-clock time."""
    if phase == "to_approach":
        if approach_distance is not None and (
                approach_distance <= approach_tolerance):
            return "advance_to_candidate"
        return "keep_navigating"

    if phase == "to_candidate":
        if (final_distance is not None and
                final_distance <= final_tolerance and
                travelled_distance >= minimum_travel):
            return "candidate_reached"
        return "keep_navigating"

    return "inactive"

