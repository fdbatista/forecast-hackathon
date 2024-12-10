import { Controller, Get } from '@nestjs/common';
import { ForecastService } from './forecast.service';

@Controller('/forecast')
export class ForecastController {

    constructor(private readonly forecastService: ForecastService) {}

    @Get()
    async forecast() {
        return await this.forecastService.forecast();
    }

}
