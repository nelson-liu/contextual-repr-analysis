#!/bin/bash
# This script is meant to be called by the "after_success" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.

set -e

if [[ "$COVERAGE" == "true" ]]; then
    # Ignore codecov failures as the codecov server is not
    # very reliable but we don't want travis to report a failure
    # in the github UI just because the coverage report failed to
    # be published.
    codecov --token=24ed68db-f2f3-4326-bdef-8a971a4023ab || echo "codecov upload failed"
fi
