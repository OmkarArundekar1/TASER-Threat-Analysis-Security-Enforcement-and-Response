"""A separate client process, for the cases a thread can't show. It holds nothing from its parent.
usage: marker_worker.py once EVENT KEY          runs one marked write, prints the outcome
       marker_worker.py race EVENT KEY GOFILE   waits for GOFILE, then does the same
       marker_worker.py open EVENT KEY          leaves a marked write uncommitted, says so, and waits to be killed
       marker_worker.py through_proxy EVENT KEY runs one marked write through the fault proxy, for a parent holding its COMMIT
"""
import json
import os
import sys
import time

import marker_lib as m

action, event_id, key = sys.argv[1:4]
m.confirm_environment()

if action == "through_proxy":
    driver = m.proxied_driver()
    print(json.dumps({"ready": True}), flush=True)
    result = m.helper(driver).write_query("d6", m.MARKED_DUPLICATE_UPDATE, m.params(event_id, key))
    print(json.dumps(m.describe(result)), flush=True)
    driver.close()
    sys.exit(0)

driver = m.direct_driver()
m.verify_disposable(driver)
if action == "open":
    session = driver.session()
    tx = session.begin_transaction()
    tx.run(m.MARKED_DUPLICATE_UPDATE, m.params(event_id, key)).consume()
    print(json.dumps({"open": True}), flush=True)
    time.sleep(120)
    sys.exit(3)
if action == "race":
    while not os.path.exists(sys.argv[4]):
        time.sleep(0.001)
result = m.helper(driver).write_query("d6", m.MARKED_DUPLICATE_UPDATE, m.params(event_id, key))
print(json.dumps(m.describe(result)), flush=True)
driver.close()
