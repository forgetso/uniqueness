'use strict'

import * as api from "../helpers/api";
import React, {Component} from 'react';

class CorrectionsList extends Component {
    constructor(props) {
        super(props);

    }

    listItems() {
        return this.props.corrections.map((correction) =>
            <li key={correction.toString()}>
                <button type="button" className="btn btn-primary btn-lg"
                        onClick={this.handleClick}>{correction}</button>
            </li>
        )
    }

    handleClick = (event) => {
        this.props.sendSpellingCorrection(this.props.response_id, event.target.innerText);
    }

    render() {
        return (<ul className="list-group">{this.listItems()}</ul>)
    }
}


export default CorrectionsList;
