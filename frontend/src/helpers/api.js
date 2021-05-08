var getRelationQuery = function () {
    return fetch(process.env.REACT_APP_API_URL + 'relationquery',
        {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                "Accept": "application/json",
                "Access-Control-Allow-Origin": "*"
            }
        }).then((res) => {
        return res.json();
    }).then((data) => {
        return data;
    });
};

var saveResponse = function (relationGroupId, relationIds, timeDurations, fingerprint, relationType) {
    /*
    relationsgroupid - the ID of the relations group: int
    relationids - the IDs of the selected relations: object {int: time_till_selection}
    fingerprint - a browser fingerprint to detect same user: string
     */
    return fetch(process.env.REACT_APP_API_URL + "relationquery",
        {
            method: 'POST',
            body: JSON.stringify({
                'relationgroupid': relationGroupId,
                'relationids': relationIds,
                'timedurations': timeDurations,
                'fingerprint': fingerprint,
                'relationtype': relationType
            }),
            headers: {
                //'X-Api-Key': API_KEY,
                'Content-Type': 'application/json',
                "Accept": "application/json",
                "Access-Control-Allow-Origin": "*"
            }
        }
    ).then(function (response) {
        return response
    });
};

exports.getRelationQuery = getRelationQuery;
exports.saveResponse = saveResponse;
