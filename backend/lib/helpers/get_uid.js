function RandomRangeInt(a, b)
{
    return a + Math.floor(Math.random() * (b + 1))
}

function getUID() {
    for (var a = (new Date).getTime().toString(36); 16 > a.length; )

        a += RandomRangeInt(0, 35).toString(36);
    return a
}

console.log(getUID())